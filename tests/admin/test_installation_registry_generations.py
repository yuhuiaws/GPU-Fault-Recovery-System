"""Reinstallation uses new registry keys, not mutable identities or lost history."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import installation_lifecycle as lifecycle_records
from gpu_fault.admin import resource_registry as registry
from gpu_fault.admin import uninstall as uninstall_module
from gpu_fault.admin.aws_cleanup_ownership import CleanupOwnership
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deploy_command import retire_site_after_uninstall
from gpu_fault.admin.site import SiteConfigError, load_site, materialized_release_config
from gpu_fault.installation_resources import InstallationResourceSnapshot
from gpu_fault.store import InMemoryStore
from scripts.release_deploy_evidence import installation_resource_registry_record
from tests.admin.test_uninstall_lifecycle import STATE, Harness


class RegistryStoreTransport:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store
        self.scopes: list[str] = []
        self.synced: list[InstallationResourceSnapshot] = []

    def __call__(self, arguments: list[str], **options: Any):
        assert arguments[0] == "kubectl", "registry transport cannot call AWS"
        if "get" in arguments:
            assert "pod" in arguments
            return subprocess.CompletedProcess(arguments, 0, "registry-cpu", "")
        assert "exec" in arguments and "registry-cpu" in arguments
        if arguments[-1] == registry.FETCH_SCRIPT:
            scope = next(
                value.split("=", 1)[1]
                for value in arguments
                if value.startswith("GPU_FAULT_INSTALLATION_SITE_ID=")
            )
            self.scopes.append(scope)
            rows = self.store.list_installation_resources(scope)
        else:
            assert arguments[-1] in {registry.SYNC_SCRIPT, registry.DIRECT_SYNC_SCRIPT}
            value = InstallationResourceSnapshot.model_validate_json(
                options["input_text"]
            )
            rows = [
                self.store.save_installation_resource(item) for item in value.resources
            ]
            self.synced.append(value)
        output = (
            str(len(rows))
            if arguments[-1] == registry.DIRECT_SYNC_SCRIPT
            else json.dumps([item.model_dump(mode="json") for item in rows])
        )
        return subprocess.CompletedProcess(arguments, 0, output, "")


def reenroll_site(harness: Harness, archive: Path) -> None:
    harness.site.source.write_bytes((archive / "site.yaml").read_bytes())
    harness.site.source.chmod(0o600)
    harness.site = load_site(harness.site.source)


def test_two_cycles_with_recreated_aws_ids_preserve_prior_generation_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    store = InMemoryStore()
    transport = RegistryStoreTransport(store)
    monkeypatch.setattr(registry, "run_command", transport)
    # Keep real registry serialization and Store identity checks under uninstall.
    monkeypatch.setattr(
        uninstall_module,
        "fetch_installation_resource_registry",
        registry.fetch_installation_resource_registry,
    )
    monkeypatch.setattr(
        uninstall_module,
        "sync_installation_resource_snapshot",
        registry.sync_installation_resource_snapshot,
    )
    runtime = {
        "nlb": {
            "arn": "arn:aws:elasticloadbalancing:us-east-1:123456789012:"
            "loadbalancer/net/fixture/old",
            "dns_name": "old.elb.example",
        }
    }
    monkeypatch.setattr(registry, "discover_runtime_resources", lambda _site: runtime)
    # Initial enrollment retains the bare legacy site scope.
    for item in harness.snapshot.resources:
        store.save_installation_resource(item)
    legacy_scope = harness.site.registry_site_id
    assert legacy_scope == harness.site.metadata_name
    uninstall_module.uninstall(harness.request(), runner=harness)
    first_state = harness.state()
    first_rows = copy.deepcopy(store.list_installation_resources(legacy_scope))
    archive = retire_site_after_uninstall(tmp_path)
    assert archive is not None
    reenroll_site(harness, archive)
    lifecycle_records.write_record(
        tmp_path / "bootstrap-state.json",
        {"site_id": harness.site.metadata_name, "resources": {}},
    )
    second_scope = harness.site.registry_site_id
    assert second_scope != legacy_scope
    assert harness.site.metadata_name == legacy_scope
    assert ":installation:" in second_scope
    # This is the real post-commit builder/sync entry point, not a warning stub.
    release_record, warning = installation_resource_registry_record(harness.site)
    assert release_record["status"] == "SYNCED" and warning is None
    built = registry.fetch_installation_resource_registry(harness.site)
    nlb = next(item for item in built.resources if item.resource_type == "nlb")
    assert nlb.resource_arn is not None and nlb.resource_arn.endswith("/old")
    assert built.site_id == second_scope
    assert store.list_installation_resources(legacy_scope) == first_rows
    # Complete the current inventory with the same topology and new AWS IDs.
    previous = {item.resource_key: item for item in built.resources}
    for item in harness.snapshot.resources:
        previous.setdefault(
            item.resource_key,
            item.model_copy(
                update={
                    "site_id": second_scope,
                    "resource_id": "new-" + item.resource_id
                    if item.resource_type == "helm_release"
                    else item.resource_id,
                }
            ),
        )
    resources = list(previous.values())
    current = InstallationResourceSnapshot(site_id=second_scope, resources=resources)
    current = current.model_copy(update={"source_sha256": current.digest()})
    registry.sync_installation_resource_snapshot(harness.site, current)
    harness.snapshot = current
    harness.existing = {item.resource_key for item in current.resources}
    harness.namespace_uid = "second-cpu-namespace"
    uninstall_module.uninstall(harness.request(), runner=harness)
    second_state = harness.state()
    cleanup = STATE.read_state(tmp_path / "uninstall/kubernetes-cleanup.json")
    assert second_state["registry_site_id"] == second_scope
    assert first_state["installation_id"] != second_state["installation_id"]
    assert cleanup["namespace_snapshots"]["cpu"]["uid"] == "second-cpu-namespace"
    assert store.list_installation_resources(legacy_scope) == first_rows
    assert (
        store.get_installation_resource(legacy_scope, "aws/nlb").resource_id
        == "nlb-test"
    )
    assert (
        store.get_installation_resource(second_scope, "aws/nlb").resource_id
        != "nlb-test"
    )
    assert harness.events.count("cleanup") == 2
    assert {legacy_scope, second_scope}.issubset(transport.scopes), (
        "each uninstall must fetch its own recorded registry scope"
    )
    second_rows = copy.deepcopy(store.list_installation_resources(second_scope))
    second_archive = retire_site_after_uninstall(tmp_path)
    assert second_archive is not None and second_archive != archive
    reenroll_site(harness, second_archive)
    lifecycle_records.write_record(
        tmp_path / "bootstrap-state.json",
        {"site_id": harness.site.metadata_name, "resources": {}},
    )
    third_scope = harness.site.registry_site_id
    assert third_scope not in {legacy_scope, second_scope}
    runtime["nlb"]["arn"] = runtime["nlb"]["arn"].removesuffix("/old") + "/new"
    runtime["nlb"]["dns_name"] = "new.elb.example"
    release_record, warning = installation_resource_registry_record(harness.site)
    assert release_record["status"] == "SYNCED" and warning is None
    assert store.list_installation_resources(legacy_scope) == first_rows
    assert store.list_installation_resources(second_scope) == second_rows
    assert (
        store.get_installation_resource(third_scope, "aws/nlb").resource_arn
        == (runtime["nlb"]["arn"])
    )
    current_nlb = store.get_installation_resource(third_scope, "aws/nlb")
    with pytest.raises(ValueError, match="identity cannot change"):
        store.save_installation_resource(
            current_nlb.model_copy(
                update={
                    "resource_arn": runtime["nlb"]["arn"].removesuffix("/new")
                    + "/unbound"
                }
            )
        )


def new_generation(harness: Harness) -> str:
    uninstall_module.uninstall(harness.request(), runner=harness)
    archive = retire_site_after_uninstall(harness.site.source.parent)
    assert archive is not None
    reenroll_site(harness, archive)
    return harness.site.registry_site_id


@pytest.mark.parametrize(
    "operation",
    [
        registry.sync_installation_resource_snapshot,
        registry.sync_installation_resource_snapshot_direct,
        registry.write_installation_resource_snapshot,
    ],
)
def test_current_registry_refuses_previous_scope_before_io(
    tmp_path, monkeypatch, operation
):
    harness = Harness(tmp_path, monkeypatch)
    new_generation(harness)
    monkeypatch.setattr(
        registry,
        "run_command",
        lambda *_args, **_kwargs: pytest.fail("old registry scope reached I/O"),
    )
    with pytest.raises(BootstrapError, match="another site"):
        operation(harness.site, harness.snapshot)


def test_cleanup_uses_generation_scope_but_keeps_logical_aws_owner(
    tmp_path, monkeypatch
):
    harness = Harness(tmp_path, monkeypatch)
    scope = new_generation(harness)
    ownership = CleanupOwnership(harness.site)
    old = next(
        item for item in harness.snapshot.resources if item.resource_type == "nlb"
    )
    current = old.model_copy(update={"site_id": scope})
    assert ownership.site_id == harness.site.metadata_name
    ownership.validate_scope(current)
    with pytest.raises(BootstrapError, match="mismatch"):
        ownership.validate_scope(old)
    with materialized_release_config(harness.site) as path:
        config = json.loads(path.read_text())
    assert config["site_name"] == harness.site.metadata_name
    assert config["registry_site_id"] == scope


@pytest.mark.parametrize("drift", ["missing", "bare", "foreign", "identifier"])
def test_generation_scope_cannot_fall_back_or_change_silently(
    tmp_path, monkeypatch, drift
):
    harness = Harness(tmp_path, monkeypatch)
    new_generation(harness)
    path = tmp_path / lifecycle_records.INSTALLATION_FILE
    record = lifecycle_records.read_record(path)
    if drift == "missing":
        path.unlink()
    else:
        if drift == "identifier":
            record["installation_id"] = "b" * 32
        else:
            record["registry_site_id"] = (
                harness.site.metadata_name if drift == "bare" else "foreign-scope"
            )
        lifecycle_records.write_record(path, record)
    with pytest.raises(SiteConfigError, match="missing|scope"):
        registry.build_installation_snapshot(harness.site, None)


def test_existing_lifecycle_enrollment_keeps_its_original_scope(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    identity = lifecycle_records.installation_for_uninstall(harness.site)
    marker = lifecycle_records.read_record(
        tmp_path / lifecycle_records.INSTALLATION_FILE
    )
    assert marker["registry_site_id"] == harness.site.metadata_name
    assert harness.site.registry_site_id == harness.site.metadata_name
    assert lifecycle_records.installation_for_uninstall(harness.site) == identity
    # Old first-generation local records remain legacy, never inferred as a new scope.
    marker.pop("registry_site_id")
    lifecycle_records.write_record(
        tmp_path / lifecycle_records.INSTALLATION_FILE, marker
    )
    assert harness.site.registry_site_id == harness.site.metadata_name


def test_current_generation_cannot_adopt_an_old_global_bootstrap_inventory(
    tmp_path, monkeypatch
):
    harness = Harness(tmp_path, monkeypatch)
    new_generation(harness)
    home = tmp_path / "home"
    old = home / ".gpu-fault/bootstrap/old/bootstrap-state.json"
    old.parent.mkdir(parents=True)
    old.write_text(
        json.dumps({"site_id": harness.site.metadata_name, "resources": {"old": {}}})
    )
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(BootstrapError, match="current installation bootstrap"):
        registry.find_bootstrap_state(harness.site)
    current = {"site_id": harness.site.metadata_name, "resources": {"current": {}}}
    lifecycle_records.write_record(tmp_path / "bootstrap-state.json", current)
    assert registry.find_bootstrap_state(harness.site) == current


def test_join_commit_and_removal_plan_select_only_current_generation(
    tmp_path, monkeypatch
):
    from gpu_fault.admin.cluster_removal_resources import target_resource_plan
    from gpu_fault.installation_lifecycle import (
        generation_registry_site_id,
        site_identity,
    )
    from tests.admin._cov95_join_support import JoinScenario

    scenario = JoinScenario(tmp_path, monkeypatch)
    installation_id = "a" * 32
    scope = generation_registry_site_id(scenario.site.metadata_name, installation_id)
    lifecycle_records.write_record(
        tmp_path / lifecycle_records.INSTALLATION_FILE,
        {
            "schema_version": 1,
            "installation_id": installation_id,
            "site_identity": site_identity(scenario.site.release_config),
            "registry_site_id": scope,
        },
    )
    store = InMemoryStore()
    for item in scenario.registry.live.resources:
        store.save_installation_resource(item)
    old_rows = copy.deepcopy(
        store.list_installation_resources(scenario.site.metadata_name)
    )
    for item in registry.build_installation_snapshot(scenario.site, None).resources:
        store.save_installation_resource(item)
    transport = RegistryStoreTransport(store)
    monkeypatch.setattr(registry, "run_command", transport)
    result = scenario.join()
    assert result["phase"] == "COMPLETED"
    current = registry.fetch_installation_resource_registry(load_site(scenario.path))
    assert current.site_id == scope
    target, waves = target_resource_plan(load_site(scenario.path), "hp-gpu-b", current)
    assert target and waves, "the joined member must have a nonempty removal plan"
    assert all(item.site_id == scope for item in target), (
        "removal must not plan resources from historical registry generations"
    )
    assert store.list_installation_resources(scenario.site.metadata_name) == old_rows
    assert (
        store.get_installation_resource(scope, "cluster/hp-gpu-b/eks").resource_id
        == "gpu-b"
    )
    old_scope = InstallationResourceSnapshot(
        site_id=scenario.site.metadata_name,
        resources=[
            item.model_copy(update={"site_id": scenario.site.metadata_name})
            for item in current.resources
        ],
    )
    old_scope = old_scope.model_copy(update={"source_sha256": old_scope.digest()})
    with pytest.raises(BootstrapError, match="another site"):
        target_resource_plan(load_site(scenario.path), "hp-gpu-b", old_scope)


def test_aurora_cleanup_keeps_physical_tag_checks_for_generation_records(
    tmp_path, monkeypatch
):
    from gpu_fault.admin import aws_commands
    from gpu_fault.admin.aws_cleanup import ResourceCleaner
    from tests.admin.test_admin_aws_cleanup_aurora import CLUSTER_ARN, AuroraAws

    harness = Harness(tmp_path, monkeypatch)
    scope = new_generation(harness)
    aws = AuroraAws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    cluster = next(
        item
        for item in harness.snapshot.resources
        if item.resource_type == "aurora_cluster"
    ).model_copy(update={"site_id": scope, "resource_arn": CLUSTER_ARN})
    cleaner = ResourceCleaner(harness.site)
    binding = cleaner.prepare_aurora_delete(
        cluster, final_snapshot_policy="skip", final_snapshot_identifier="unused"
    )
    assert binding["db_cluster_resource_id"] == "cluster-incarnation-a"
    assert aws.mutations == []
    aws.database["TagList"] = [{"Key": "gpu-fault:site-id", "Value": scope}]
    with pytest.raises(BootstrapError):
        cleaner.prepare_aurora_delete(
            cluster, final_snapshot_policy="skip", final_snapshot_identifier="unused"
        )
    assert aws.mutations == []


def test_registry_api_keeps_current_and_historical_scopes_separate():
    import asyncio

    from gpu_fault.app import ApplicationContext
    from gpu_fault.installation_lifecycle import generation_registry_site_id
    from tests._builders import asgi_client
    from tests.admin.test_registry_transport import snapshot

    token = "registry-generation-fixture-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    context.regional_mode = True
    previous = snapshot()
    scope = generation_registry_site_id(previous.site_id, "a" * 32)
    current = InstallationResourceSnapshot(
        site_id=scope,
        resources=[
            item.model_copy(update={"site_id": scope, "resource_id": "new-resource"})
            for item in previous.resources
        ],
    )

    async def scenario():
        async with asgi_client(context) as client:
            headers = {"X-GPU-Fault-Execution-Token": token}
            for value in (previous, current):
                response = await client.post(
                    "/v1/installation-resources/sync",
                    headers=headers,
                    json=value.model_dump(mode="json"),
                )
                assert response.status_code == 200
            for value in (previous, current):
                response = await client.get(
                    "/v1/installation-resources",
                    params={"site_id": value.site_id},
                    headers=headers,
                )
                assert response.status_code == 200
                assert response.json() == [
                    item.model_dump(mode="json") for item in value.resources
                ]
            denied = await client.get(
                "/v1/installation-resources", params={"site_id": scope}
            )
            assert denied.status_code == 403

    asyncio.run(scenario())
