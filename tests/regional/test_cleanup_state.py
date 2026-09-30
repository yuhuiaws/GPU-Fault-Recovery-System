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
    for phase in MODULE.required_phases(document):
        MODULE.transition(document, phase=phase, status="COMPLETED", message="fixture")
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
    for phase in MODULE.required_phases(document):
        MODULE.transition(document, phase=phase, status="COMPLETED", message="fixture")
    MODULE.transition(
        document,
        phase="CLEANUP_COMPLETED",
        status="COMPLETED",
        message="cleanup complete",
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


def test_cleanup_resume_binds_selected_cluster_and_original_request(
    tmp_path: Path,
) -> None:
    config, inventory = inputs(tmp_path)
    value = json.loads(config.read_text())
    value["clusters"].append({"cluster_id": "gpu-b", "context": "gpu-b-context"})
    config.write_text(json.dumps(value))
    path = tmp_path / "cleanup.json"
    document = MODULE.initialize(
        path,
        config_path=config,
        inventory_path=inventory,
        scope="gpu",
        mode="clean",
        node_mode="uninstall",
        cluster_ids=["gpu-b"],
    )
    assert document["targets"]["clusters"] == [
        {"cluster_id": "gpu-b", "context": "gpu-b-context"}
    ]
    MODULE.validate_request(
        document,
        config_path=config,
        scope="gpu",
        mode="clean",
        node_mode="uninstall",
        cluster_ids=["gpu-b"],
    )
    with pytest.raises(MODULE.CleanupStateError, match="differs on targets"):
        MODULE.validate_request(
            document,
            config_path=config,
            scope="gpu",
            mode="clean",
            node_mode="uninstall",
            cluster_ids=["gpu-a"],
        )
    value["namespace"] = "different"
    config.write_text(json.dumps(value))
    with pytest.raises(MODULE.CleanupStateError, match="config_sha256"):
        MODULE.validate_request(
            document,
            config_path=config,
            scope="gpu",
            mode="clean",
            node_mode="uninstall",
            cluster_ids=["gpu-b"],
        )


@pytest.mark.parametrize("phase", ["CLEANUP_COMPLETED", "READY_TO_DELETE_AURORA"])
def test_cleanup_cannot_skip_drain_and_node_checkpoints(
    tmp_path: Path, phase: str
) -> None:
    config, inventory = inputs(tmp_path)
    document = MODULE.initialize(
        tmp_path / "cleanup.json",
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    with pytest.raises(MODULE.CleanupStateError, match="requires completed"):
        MODULE.transition(
            document, phase=phase, status="COMPLETED", message="too early"
        )


@pytest.mark.parametrize("legacy_version", [1, True, 2.0, "2"])
def test_cleanup_rejects_legacy_or_invalid_phase_schema(
    tmp_path: Path, legacy_version: object
) -> None:
    config, inventory = inputs(tmp_path)
    document = MODULE.initialize(
        tmp_path / "cleanup.json",
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    assert document["schema_version"] == 2, "new ordering needs an explicit schema"
    document["schema_version"] = legacy_version
    document["content_sha256"] = MODULE.content_digest(document)
    with pytest.raises(
        MODULE.CleanupStateError, match="phase order requires reconciliation"
    ):
        MODULE.verify_document(document)


def test_cleanup_records_node_shutdown_after_consumers_and_executors(
    tmp_path: Path,
) -> None:
    config, inventory = inputs(tmp_path)
    document = MODULE.initialize(
        tmp_path / "cleanup.json",
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    for phase in MODULE.required_phases(document):
        MODULE.transition(document, phase=phase, status="COMPLETED", message=phase)
    completed = MODULE.completed_phases(document)
    assert completed.index("CLUSTERS_DRAINING") < completed.index(
        "GPU_DATA_PLANE_SOURCES_STOPPED"
    ), "the producer stopped before registry admission closed"
    assert completed.index("QUEUES_DRAINED") < completed.index(
        "CONTROL_CONSUMERS_STOPPED"
    ), "consumers were stopped before the drain"
    assert completed.index("CONTROL_CONSUMERS_STOPPED") < completed.index(
        "INGRESS_STOPPED"
    ), "ingress stopped while consumers could still dispatch"
    assert completed.index("INGRESS_STOPPED") < completed.index(
        "GPU_EXECUTORS_STOPPED"
    ), "Executor shutdown preceded CPU shutdown"
    assert completed.index("CONTROL_CONSUMERS_STOPPED") < completed.index(
        "GPU_EXECUTORS_STOPPED"
    ), "the executor stopped before its consumers"
    assert completed.index("GPU_EXECUTORS_STOPPED") < completed.index(
        "NODE_RUNTIMES_STOPPED"
    ), "node shutdown could still cause new claims"
    with pytest.raises(MODULE.CleanupStateError, match="cannot move backwards"):
        MODULE.transition(
            document,
            phase="GPU_EXECUTORS_STOPPED",
            status="COMPLETED",
            message="old order",
        )


def test_cleanup_cannot_import_fleet_authority_from_another_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    before = path.read_bytes()
    monkeypatch.setattr(
        "sys.argv",
        [
            "cleanup_state.py",
            "reuse-fleet",
            "--path",
            str(path),
            "--from",
            str(tmp_path / "foreign-missing.json"),
        ],
    )
    with pytest.raises(
        MODULE.CleanupStateError, match="resume the original bound journal"
    ):
        MODULE.main()
    assert path.read_bytes() == before, (
        "refused fleet import changed the original journal"
    )


@pytest.mark.parametrize("proof", ["missing", "ingress-first", "incoming-main"])
def test_schema_v2_does_not_authorize_a_different_cleanup_phase_order(
    tmp_path: Path, proof: str
) -> None:
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
    assert document["schema_version"] == 2
    assert document["phase_order"] == list(MODULE.PHASES)
    if proof == "missing":
        document.pop("phase_order")
    elif proof == "ingress-first":
        order = document["phase_order"]
        order.remove("CLUSTERS_DRAINING")
        order.remove("INGRESS_STOPPED")
        order.insert(order.index("QUEUES_DRAINED"), "INGRESS_STOPPED")
    else:
        order = document["phase_order"]
        order.remove("NODE_RUNTIMES_STOPPED")
        order.insert(order.index("INGRESS_STOPPED"), "UNCLAIMABLE_WORK_ABANDONED")
    MODULE.atomic_write(path, document)
    before = path.read_bytes()

    with pytest.raises(MODULE.CleanupStateError, match="phase order.*reconciliation"):
        MODULE.read_state(path)
    with pytest.raises(MODULE.CleanupStateError, match="phase order.*reconciliation"):
        MODULE.transition(
            document,
            phase="CLUSTERS_DRAINING",
            status="IN_PROGRESS",
            message="old in-memory journal",
        )

    assert path.read_bytes() == before, "incompatible evidence was rewritten"


def test_scope_all_cannot_complete_without_the_registry_drain_phase(
    tmp_path: Path,
) -> None:
    config, inventory = inputs(tmp_path)
    document = MODULE.initialize(
        tmp_path / "cleanup.json",
        config_path=config,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    for phase in MODULE.required_phases(document):
        if phase != "CLUSTERS_DRAINING":
            MODULE.transition(document, phase=phase, status="COMPLETED", message=phase)
    with pytest.raises(MODULE.CleanupStateError, match="completed lifecycle"):
        MODULE.transition(
            document, phase="CLEANUP_COMPLETED", status="COMPLETED", message="missing"
        )


def test_phase_order_drains_the_registry_before_ingress_stops() -> None:
    """Live uninstall (2026-09-15): the drain waited under a stopped ingress, so
    leases that could no longer renew expired, the still-running worker failed
    their workflows and minted drain successors, and the queue never reached
    zero. The clusters are published DRAINING first, the drain waits with
    ingress and the consumers up, the consumers stop, and only then does
    ingress stop; leftovers fail the run closed instead of being bulk-failed."""

    assert MODULE.PHASES == (
        "PREFLIGHT",
        "CLUSTERS_DRAINING",
        "GPU_DATA_PLANE_SOURCES_STOPPED",
        "QUEUES_DRAINED",
        "CONTROL_CONSUMERS_STOPPED",
        "INGRESS_STOPPED",
        "GPU_EXECUTORS_STOPPED",
        "NODE_RUNTIMES_STOPPED",
        "CPU_AUXILIARIES_STOPPED",
        "APPLICATION_OBJECTS_DELETED",
        "NAMESPACES_DELETED",
        "CLEANUP_COMPLETED",
        "READY_TO_DELETE_AURORA",
        "AURORA_DELETED",
    )
    assert "UNCLAIMABLE_WORK_ABANDONED" not in MODULE.PHASES, (
        "the cleanup never bulk-fails leftover work; undrained rows fail closed"
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


def test_a_record_of_the_earlier_order_is_neither_continued_nor_a_fleet_source(
    tmp_path: Path,
) -> None:
    """A record written under the ingress-first order (schema 1) must never be
    continued -- one history would mix two phase orders -- and it is not a
    fleet-snapshot authority for a new journal either: the new run starts a
    fresh schema-2 record and exports the fleet from the live control plane."""

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
    document.pop("phase_order")
    document["phase"] = "INGRESS_STOPPED"
    document["status"] = "FAILED"
    MODULE.attach_fleet_snapshot(document, [{"cluster_id": "gpu-a", "node_id": "n"}])
    MODULE.atomic_write(earlier, document)
    before = earlier.read_bytes()

    with pytest.raises(MODULE.CleanupStateError, match="reconciliation"):
        MODULE.read_state(earlier)
    with pytest.raises(MODULE.CleanupStateError, match="reconciliation"):
        MODULE.transition(
            document, phase="QUEUES_DRAINED", status="IN_PROGRESS", message="continue"
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
    fresh_before = fresh.read_bytes()
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
    assert completed.returncode != 0, completed.stdout
    assert "resume the original bound journal" in completed.stderr, completed.stderr
    assert fresh.read_bytes() == fresh_before, (
        "refused fleet import changed the journal"
    )
    assert earlier.read_bytes() == before, "the earlier record was rewritten"
    assert MODULE.read_state(fresh)["fleet_snapshot"] is None
