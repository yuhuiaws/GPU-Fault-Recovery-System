"""Validate native uninstall evidence without replacing its journal or policies."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.aws_cleanup import is_aurora_resource
from gpu_fault.admin.resource_registry import load_installation_resource_snapshot
from gpu_fault.admin.site import materialized_release_config
from gpu_fault.admin.uninstall import CPU_CLUSTER_BOUND_RESOURCE_KEYS, UNINSTALL_PHASES
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceSnapshot,
)
from gpu_fault.installation_resources import (
    InstallationResourceDeletePolicy as Policy,
)
from gpu_fault.installation_resources import (
    InstallationResourceOwnership as Ownership,
)
from gpu_fault.installation_resources import (
    InstallationResourceStatus as Status,
)
from scripts.e2e.regional.boot032_contract import (
    CASE_ID,
    Settings,
    UninstallCaseError,
    checked_path,
    cluster_specs,
    mapping,
    read_document,
    require,
)
from scripts.e2e.regional.boot032_fleet import verify_fleet_receipt
from scripts.e2e.regional.live_driver_guard import details_sha256

CLEANUP = import_module("deploy.control-plane.tools.cleanup_state")
CASE_PHASES = frozenset(
    {"STARTED", "RESTART_REQUIRED", "RESUMING", "COMPLETED", "FAILED", "BLOCKED"}
)


def fixture_policy(resource: InstallationResource) -> Policy:
    """Assert the fixed full-retirement contract, never choose mutation targets."""
    if resource.resource_type in {"gpu_eks", "gpu_hyperpod"}:
        return Policy.PRESERVE
    if (
        resource.resource_type in {"cpu_eks", "cpu_hyperpod"}
        or resource.resource_key in CPU_CLUSTER_BOUND_RESOURCE_KEYS
        or is_aurora_resource(resource)
    ):
        return Policy.DELETE
    return resource.delete_policy


def expected_policies(snapshot: InstallationResourceSnapshot) -> dict[str, str]:
    return {
        item.resource_key: fixture_policy(item).value for item in snapshot.resources
    }


def snapshot_identities(snapshot: InstallationResourceSnapshot) -> dict[str, Any]:
    return {
        item.resource_key: {
            "identity": list(item.immutable_identity()),
            "attributes": item.attributes,
        }
        for item in snapshot.resources
    }


def native_state(settings: Settings) -> dict[str, Any] | None:
    path = settings.native_dir / "state.json"
    require(not path.is_symlink(), "native uninstall journal is a symbolic link")
    if not path.exists():
        return None
    value = read_document(path)
    require(
        value.get("phase") in UNINSTALL_PHASES
        and value.get("site_id") == settings.target.metadata_name
        and value.get("site_sha256") == settings.target.source_sha256
        and value.get("cpu_disposition") == "delete"
        and value.get("final_snapshot_policy") == "retain"
        and value.get("reset_database") is False,
        "native uninstall journal differs from the approved retirement",
    )
    require(
        "supervision_lost" not in value, "native uninstall lost process supervision"
    )
    return value


def reached(state: dict[str, Any], phase: str) -> bool:
    return UNINSTALL_PHASES.index(state["phase"]) >= UNINSTALL_PHASES.index(phase)


def required_native_state(settings: Settings) -> dict[str, Any]:
    return mapping(native_state(settings), "native uninstall journal is missing")


def native_snapshot(settings: Settings, name: str) -> InstallationResourceSnapshot:
    path = checked_path(settings.native_dir / name)
    checked_path(path.with_suffix(path.suffix + ".sha256"))
    return load_installation_resource_snapshot(path)


def cleanup_receipt(settings: Settings, binding: dict[str, Any]) -> dict[str, Any]:
    path = checked_path(settings.native_dir / "kubernetes-cleanup.json")
    document = mapping(
        CLEANUP.read_state(path), "native cleanup receipt is not an object"
    )
    with materialized_release_config(settings.target) as config:
        CLEANUP.validate_request(
            document,
            config_path=config,
            scope="all",
            mode="reset",
            node_mode="uninstall",
            cluster_ids=[],
        )
    require(
        details_sha256(
            mapping(document.get("inventory_snapshot"), "cleanup inventory is missing")
        )
        == binding["target"]["inventory_sha256"],
        "native cleanup inventory differs from the approved installed inventory",
    )
    contexts = {
        "cpu" if item["plane"] == "cpu" else "gpu:" + item["context"]: item
        for item in cluster_specs(settings.target)
    }
    require(
        set(document.get("cluster_uids", {})) == set(contexts),
        "native cleanup cluster inventory is incomplete",
    )
    for key, spec in contexts.items():
        expected = binding["target"]["clusters"][spec["context"]]
        require(
            document["cluster_uids"][key] == expected["cluster_uid"],
            "native cleanup Kubernetes cluster identity changed",
        )
        snapshot = document.get("namespace_snapshots", {}).get(key)
        require(
            isinstance(snapshot, dict)
            and snapshot.get("uid") == expected["namespace_uid"],
            "native cleanup namespace identity changed",
        )
        namespace_objects(
            settings, binding, spec, mapping(snapshot, "namespace snapshot is missing")
        )
    return document


def namespace_objects(
    settings: Settings,
    binding: dict[str, Any],
    spec: dict[str, str],
    snapshot: dict[str, Any],
) -> None:
    objects = snapshot.get("objects")
    if not isinstance(objects, list):
        raise UninstallCaseError("namespace object identity inventory is missing")
    identities = {}
    for item in objects:
        require(
            isinstance(item, list)
            and len(item) == 4
            and all(isinstance(value, str) and value for value in item),
            "namespace object identity is malformed",
        )
        key = item[1].lower(), item[2]
        require(key not in identities, "namespace object identity is duplicated")
        identities[key] = item[3]
    for row in binding["target"]["resource_uids"][spec["context"]]:
        scope, kind, namespace, name = row["identity"]
        if (
            scope == "namespaced"
            and namespace == settings.target.release_config["namespace"]
        ):
            require(
                identities.get((kind.lower(), name)) == row["uid"],
                "native namespace object identity differs from approved inventory",
            )


def cleanup_complete(settings: Settings, binding: dict[str, Any]) -> dict[str, Any]:
    document = cleanup_receipt(settings, binding)
    verify_fleet_receipt(settings, binding, document)
    require(
        document.get("phase") == "CLEANUP_COMPLETED"
        and document.get("status") == "COMPLETED",
        "native Kubernetes cleanup is incomplete",
    )
    expected = {"gpu:" + item["context"] for item in cluster_specs(settings.target)[1:]}
    require(
        set(document.get("node_targets", {})) == expected
        and set(document.get("node_cleanup", {})) == expected,
        "native node cleanup inventory is incomplete",
    )
    for context in expected:
        targets = document["node_targets"][context]
        record = document["node_cleanup"][context]
        require(
            isinstance(targets, dict)
            and targets
            and all(isinstance(value, str) and value for value in targets.values())
            and isinstance(record, dict)
            and record.get("status") == "REMOVED"
            and isinstance(record.get("uid"), str)
            and record["uid"],
            "native node cleanup does not prove owned targets and temporary resource removal",
        )
    return document


def export_receipts(
    settings: Settings,
    binding: dict[str, Any],
    state: dict[str, Any],
) -> InstallationResourceSnapshot:
    approved = InstallationResourceSnapshot.model_validate(
        binding["target"]["resources"]
    )
    approved.require_source_binding()
    before = native_snapshot(settings, "installation-resources-before.json")
    plan = native_snapshot(settings, "installation-resources-delete-plan.json")
    require(before.digest() == approved.digest(), "native exported registry changed")
    require(
        snapshot_identities(plan) == snapshot_identities(before),
        "native deletion plan changes resource identity or ownership",
    )
    policies = expected_policies(before)
    require(
        state.get("registry_sha256") == before.digest()
        and state.get("effective_policies") == policies,
        "native authorized registry or retirement policies changed",
    )
    for item in plan.resources:
        expected = (
            Status.ACTIVE
            if policies[item.resource_key] == Policy.PRESERVE
            else Status.DELETE_PENDING
        )
        require(item.status is expected, "native deletion plan status is inconsistent")
    return before


def pause_proof(settings: Settings, binding: dict[str, Any]) -> dict[str, Any]:
    state = required_native_state(settings)
    require(
        state["phase"] == "REGISTRY_EXPORTED",
        "controlled pause is not at the exported pre-AWS cleanup boundary",
    )
    before = export_receipts(settings, binding, state)
    cleanup = cleanup_complete(settings, binding)
    cpu = binding["target"]["clusters"]["cpu"]
    require(
        state.get("cpu_binding")
        == {
            "cpu_eks_created_at": cpu["eks_created_at"],
            "cpu_hyperpod_arn": cpu["hyperpod_arn"],
        },
        "native CPU incarnation checkpoint changed",
    )
    return {
        "registry_sha256": before.digest(),
        "cleanup_run_id": cleanup["run_id"],
        "cleanup_identity_sha256": cleanup_identity(cleanup),
        "native_cpu_binding": state["cpu_binding"],
        "native_aurora_binding": state["aurora_binding"],
        "final_snapshot_identifier": state["final_snapshot_identifier"],
    }


def cleanup_identity(cleanup: dict[str, Any]) -> str:
    return details_sha256(
        {
            key: cleanup[key]
            for key in (
                "config_sha256",
                "inventory_sha256",
                "cluster_uids",
                "namespace_snapshots",
                "node_targets",
                "node_cleanup",
                "fleet_snapshot_sha256",
            )
        }
    )


def final_receipts(
    settings: Settings,
    binding: dict[str, Any],
    pause: dict[str, Any],
) -> tuple[InstallationResourceSnapshot, dict[str, Any]]:
    state = required_native_state(settings)
    require(state["phase"] == "COMPLETED", "native uninstall is not completed")
    before = export_receipts(settings, binding, state)
    cleanup = cleanup_complete(settings, binding)
    pre = native_snapshot(settings, "installation-resources-pre-aurora-delete.json")
    final = native_snapshot(settings, "installation-resources-final.json")
    require(
        state.get("cleanup_sha256") == cleanup["content_sha256"]
        and state.get("pre_aurora_sha256") == pre.digest()
        and state.get("final_registry_sha256") == final.digest()
        and type(state.get("delete_policy_residuals")) is int
        and state["delete_policy_residuals"] == 0,
        "native terminal receipt digests or residual count are unproved",
    )
    require(
        snapshot_identities(pre) == snapshot_identities(before)
        and set(item.resource_key for item in final.resources)
        == {*snapshot_identities(before), "aws/aurora/final-snapshot"},
        "native final inventory omitted, added or replaced resources",
    )
    for snapshot in (pre, final):
        by_key = {item.resource_key: item for item in before.resources}
        for item in snapshot.resources:
            if item.resource_key == "aws/aurora/final-snapshot":
                continue
            require(
                item.immutable_identity()
                == by_key[item.resource_key].immutable_identity()
                and item.attributes == by_key[item.resource_key].attributes,
                "uninstall changed immutable resource identity or ownership",
            )
            expected = {
                Policy.DELETE: Status.DELETED,
                Policy.DETACH: Status.DETACHED,
                Policy.PRESERVE: Status.PRESERVED,
            }[fixture_policy(item)]
            if snapshot is pre and is_aurora_resource(item):
                expected = Status.DELETE_PENDING
            require(
                item.status is expected,
                "uninstall resource has an unexpected terminal policy state",
            )
    require(
        cleanup["run_id"] == pause["cleanup_run_id"]
        and cleanup_identity(cleanup) == pause["cleanup_identity_sha256"]
        and state["cpu_binding"] == pause["native_cpu_binding"]
        and state["aurora_binding"] == pause["native_aurora_binding"]
        and state["final_snapshot_identifier"] == pause["final_snapshot_identifier"],
        "resumed uninstall did not preserve its original incarnation checkpoints",
    )
    retained = next(
        item
        for item in final.resources
        if item.resource_key == "aws/aurora/final-snapshot"
    )
    require(
        retained.resource_type == "rds_snapshot"
        and retained.resource_id == state["final_snapshot_identifier"]
        and retained.ownership is Ownership.CREATED
        and retained.delete_policy is Policy.PRESERVE
        and retained.status is Status.PRESERVED
        and retained.attributes.get("db_cluster_resource_id")
        == state["aurora_binding"].get("db_cluster_resource_id")
        and bool(retained.attributes.get("db_cluster_resource_id"))
        and retained.attributes.get("cluster_id")
        == settings.target.release_config["health"]["aurora_cluster_id"],
        "retained final snapshot is not bound to the original database",
    )
    return final, cleanup


def read_case(path: Path) -> dict[str, Any] | None:
    require(not path.is_symlink(), "case journal is a symbolic link")
    if not path.exists():
        return None
    value = read_document(path)
    require(
        type(value.get("schema_version")) is int
        and value["schema_version"] == 1
        and value.get("case_id") == CASE_ID
        and value.get("phase") in CASE_PHASES
        and isinstance(value.get("events"), list)
        and value.get("content_sha256")
        == details_sha256(
            {key: item for key, item in value.items() if key != "content_sha256"}
        ),
        "case journal identity or integrity is invalid",
    )
    return value


def save_case(path: Path, value: dict[str, Any], phase: str, **updates: Any) -> None:
    require(phase in CASE_PHASES, "case phase is invalid")
    value.update(updates, phase=phase)
    value.setdefault("events", []).append(
        {
            "phase": phase,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    value["content_sha256"] = details_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    write_json_atomic(path, value)
