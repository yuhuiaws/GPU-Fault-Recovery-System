from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from gpu_fault import node_installer_reconciler

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = (
    ROOT / "deploy" / "control-plane" / "regional" / "cleanup-inventory.json"
)
GENERATOR = ROOT / "scripts" / "generate-cleanup-inventory.py"


def inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def manifest_resources(section: dict) -> set[tuple[str, str]]:
    resources: set[tuple[str, str]] = set()
    for pattern in section["manifest_globs"]:
        paths = sorted(ROOT.glob(pattern))
        assert paths, f"cleanup inventory manifest glob matched nothing: {pattern}"
        for path in paths:
            for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
                if not isinstance(document, dict):
                    continue
                kind = document.get("kind")
                name = (document.get("metadata") or {}).get("name")
                if kind and name:
                    resources.add((kind.lower(), name))
    return resources


def registered_resources(section: dict) -> set[tuple[str, str]]:
    return {(resource["kind"], resource["name"]) for resource in section["resources"]}


def test_cleanup_inventory_covers_production_manifests_bidirectionally() -> None:
    document = inventory()

    assert document["schema_version"] == 1
    assert document["generated_by"] == ("scripts/generate-cleanup-inventory.py")
    for plane in ("cpu", "gpu"):
        section = document[plane]
        actual = manifest_resources(section)
        registered = registered_resources(section)
        implicit = {kind.lower() for kind in section["implicit_namespace_kinds"]}
        unregistered = {
            resource for resource in actual - registered if resource[0] not in implicit
        }
        stale = registered - actual
        assert not unregistered, (
            f"{plane} manifests lack cleanup entries: {unregistered}"
        )
        assert not stale, f"{plane} cleanup entries lack manifests: {stale}"


def test_cleanup_inventory_declares_order_and_cluster_scope() -> None:
    document = inventory()
    cpu = document["cpu"]["resources"]
    gpu = document["gpu"]["resources"]

    assert any(
        item["kind"] == "deployment" and item["phase"] == "ingress" for item in cpu
    ), "CPU cleanup inventory has no ingress deployment"
    assert any(
        item["kind"] == "deployment" and item["phase"] == "consumer" for item in cpu
    ), "CPU cleanup inventory has no consumer deployment"
    assert any(
        item["kind"] == "deployment" and item["phase"] == "producer" for item in gpu
    ), "GPU cleanup inventory has no producer deployment"
    assert any(
        item["kind"] == "deployment" and item["phase"] == "executor" for item in gpu
    ), "GPU cleanup inventory has no executor deployment"
    for item in gpu:
        if item["scope"] == "cluster":
            assert item["clean"] == "delete", item
    assert (
        "gpu-fault.io/gpu-plugin-restart-operation"
        in document["gpu"]["node_annotations"]
    )
    assert document["gpu"]["node_labels"] == ["gpu-fault.io/spare"]


def test_cleanup_script_consumes_registry_instead_of_workload_literals() -> None:
    script = (
        ROOT / "deploy/control-plane/regional/prepare-clean-redeploy.sh"
    ).read_text(encoding="utf-8")
    document = inventory()

    assert "cleanup-inventory.json" in script
    for plane in ("cpu", "gpu"):
        for resource in document[plane]["resources"]:
            if resource["kind"] in {"deployment", "daemonset", "cronjob"}:
                assert resource["name"] not in script, resource


def test_node_uninstall_uses_recorded_units_with_legacy_discovery() -> None:
    installer = (ROOT / "deploy/node/install-gpu-fault-collector.sh").read_text(
        encoding="utf-8"
    )
    uninstaller = (ROOT / "deploy/node/uninstall-gpu-fault-collector.sh").read_text(
        encoding="utf-8"
    )

    assert "/opt/gpu-fault/installed-units.txt" in installer
    assert "/opt/gpu-fault/installed-units.txt" in uninstaller
    assert "gpu-fault-*.service" in uninstaller
    assert "gpu-fault-*.timer" in uninstaller


def test_cleanup_inventory_generator_is_current() -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_cleanup_inventory_names_every_installer_annotation() -> None:
    """Final review M3: the inventory and the reconciler share one list.

    ``prepare-clean-redeploy.sh`` clears the node annotations the inventory
    names; a name the reconciler writes but the inventory lacks survives a
    clean redeploy, and ``installer-attempts`` carrying over means an hour of
    inherited backoff.
    """

    written = {
        value
        for name, value in vars(node_installer_reconciler).items()
        if name.startswith("INSTALLER_")
        and name.endswith("_ANNOTATION")
        and isinstance(value, str)
    }
    listed = set(inventory()["gpu"]["node_annotations"])
    missing = written - listed
    assert not missing, (
        f"cleanup inventory lacks installer annotations: {sorted(missing)}"
    )
    assert set(node_installer_reconciler.INSTALLER_NODE_ANNOTATIONS) == written, (
        "the reconciler's own list must be the complete one, so both consumers "
        "can derive from it"
    )
