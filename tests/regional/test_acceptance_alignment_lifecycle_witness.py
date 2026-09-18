from __future__ import annotations

import copy
from datetime import UTC, datetime

import pytest

from gpu_fault.schema_migrations import POSTGRES_SCHEMA_MIGRATIONS
from gpu_fault_release.regional_deployment_inventory import CPU_RUNTIME_DEPLOYMENTS
from scripts.e2e.regional.boot024_epoch_preflight import isolated_epoch_scope
from scripts.e2e.regional.boot_membership_observation import (
    membership_transition_errors,
)
from scripts.e2e.regional.boot_store_lifecycle_evidence import transition_errors


def test_membership_accepts_only_the_map_bound_worker_rollout():
    before = {
        "publication": {"map_sha256": "old"},
        "deployments": {
            name: {"uid": name, "generation": 1} for name in CPU_RUNTIME_DEPLOYMENTS
        },
    }
    after = copy.deepcopy(before)
    after["publication"]["map_sha256"] = "new"
    assert membership_transition_errors(before, after) == [
        "new failure-domain map did not roll the worker"
    ]
    after["deployments"]["gpu-fault-control-worker"]["generation"] = 2
    assert membership_transition_errors(before, after) == []
    after["deployments"]["gpu-fault-api-ha"]["generation"] = 2
    assert membership_transition_errors(before, after), (
        "failure-domain map publication must not accept an unrelated API rollout"
    )


@pytest.mark.parametrize("overlap", ["none", "cpu", "gpu", "missing"])
def test_epoch_scope_does_not_relabel_protected_predecessor(overlap):
    prefix = "arn:aws:eks:us-east-1:123456789012:cluster/"
    lifecycle = {
        "cpu_eks_arn": prefix + "isolated-cpu",
        "clusters": [{"eks_cluster_arn": prefix + "isolated-gpu"}],
    }
    protected = {
        "cpu_eks_arn": prefix + "protected-cpu",
        "clusters": [{"eks_cluster_arn": prefix + "protected-gpu"}],
    }
    if overlap == "cpu":
        lifecycle["cpu_eks_arn"] = protected["cpu_eks_arn"]
    if overlap == "gpu":
        lifecycle["clusters"] = protected["clusters"]
    if overlap == "missing":
        lifecycle["clusters"] = []
    if overlap == "none":
        scope = isolated_epoch_scope(lifecycle, protected)
        assert scope["lifecycle_cpu_eks_arn"] != scope["protected_cpu_eks_arn"]
    else:
        with pytest.raises(ValueError):
            isolated_epoch_scope(lifecycle, protected)


def documents():
    before = {
        "report_type": "boot-store-lifecycle-observation",
        "state_dir_sha256": "state",
        "cpu_eks_arn": "cpu",
        "namespace_uid": "namespace-old",
        "database": {"cluster_resource_id": "original"},
        "observed_at": datetime.now(UTC).isoformat(),
        "store": {
            "schema_version": 17,
            "migrations": [
                [item.version, item.name, item.checksum]
                for item in POSTGRES_SCHEMA_MIGRATIONS
                if item.version <= 17
            ],
            "records": {"workflow/history": "unchanged"},
        },
    }
    after = copy.deepcopy(before)
    after.update(
        namespace_uid="namespace-new",
        live_jobs=[],
        installation={
            "installation_id": "new",
            "retained_uninstall": {"previous_installation_id": "old"},
        },
        release_state={
            "phase": "complete",
            "database_schema_version": 18,
            "transaction_committed": True,
            "completed_phases": ["schema-ready"],
            "schema_change_acceptance": {
                "mode": "snapshot",
                "database_schema_version": 18,
                "snapshot_id": "snapshot-before",
                "snapshot_status": "available",
            },
            "retained_database_origin": {
                "database_state": "initialized",
                "safe": True,
                "schema_ensure_required": True,
                "retained_database_handoff": {"previous_installation_id": "old"},
            },
            "bootstrap_store_safety": {
                "schema_version": 17,
                "safe": True,
                "schema_ensure_required": True,
            },
            "aurora_prerequisite_repair": {
                "jobs": {
                    "job": {
                        "name": "job",
                        "uid": "job-id",
                        "owner_uid": "cron-id",
                        "status": "REMOVED",
                    }
                }
            },
        },
    )
    after["store"]["schema_version"] = 18
    after["store"]["migrations"].append(
        [
            POSTGRES_SCHEMA_MIGRATIONS[-1].version,
            POSTGRES_SCHEMA_MIGRATIONS[-1].name,
            POSTGRES_SCHEMA_MIGRATIONS[-1].checksum,
        ]
    )
    uninstall = {
        "phase": "COMPLETED",
        "cpu_disposition": "keep",
        "reset_database": False,
        "installation_id": "old",
        "retained_database": before["database"],
    }
    return before, after, uninstall


@pytest.mark.parametrize("mode", ["schema", "fail-forward", "reinstall"])
def test_lifecycle_proof_requires_physical_transition_and_original_history(mode):
    before, after, uninstall = documents()
    if mode != "reinstall":
        after["namespace_uid"] = before["namespace_uid"]
    if mode == "fail-forward":
        after["release_state"].update(phase="failed", transaction_committed=False)
    assert transition_errors(before, after, mode=mode, uninstall=uninstall) == []
    after["database"] = {"cluster_resource_id": "replacement"}
    assert transition_errors(before, after, mode=mode, uninstall=uninstall), (
        "lifecycle proof must reject replacement of the retained database"
    )


@pytest.mark.parametrize(
    "damage",
    [
        "old-namespace",
        "empty-origin",
        "no-job",
        "orphan-job",
        "live-job",
        "lost-history",
        "reset",
        "old-installation",
        "old-schema",
    ],
)
def test_retained_reinstall_does_not_accept_a_reset_or_incomplete_cleanup(damage):
    before, after, uninstall = documents()
    if damage == "old-namespace":
        after["namespace_uid"] = before["namespace_uid"]
    elif damage == "empty-origin":
        after["release_state"]["bootstrap_database_origin"] = {}
    elif damage == "no-job":
        after["release_state"]["aurora_prerequisite_repair"] = {}
    elif damage == "orphan-job":
        after["release_state"]["aurora_prerequisite_repair"]["jobs"]["job"][
            "owner_uid"
        ] = None
    elif damage == "live-job":
        after["live_jobs"] = [{"name": "job", "uid": "job-id"}]
    elif damage == "lost-history":
        after["store"]["records"] = {}
    elif damage == "reset":
        uninstall["reset_database"] = True
    elif damage == "old-installation":
        after["installation"]["installation_id"] = "old"
    else:
        after["store"]["schema_version"] = 16
    assert transition_errors(before, after, mode="reinstall", uninstall=uninstall), (
        "retained reinstall must reject incomplete or contradictory lifecycle proof"
    )
