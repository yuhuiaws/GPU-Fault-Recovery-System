from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.admin import bootstrap_services
from gpu_fault.admin.bootstrap_common import BootstrapError, BootstrapMutationRequired
from gpu_fault.admin.node_key_custody_admin import (
    CustodyPreparationRequired,
    CustodyReconciliationRequired,
    bootstrap_custody_profile,
)
from gpu_fault.admin.node_key_custody_crypto import parse
from gpu_fault.admin.node_key_custody_models import Chain, canonical, statement_sha256
from tests.admin._node_key_custody_admin_support import AdminWorld
from tests.admin.test_node_key_custody_admin import provision
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    world = AdminWorld(tmp_path, monkeypatch)
    world.configure()
    world.access()
    world.release_ready = True
    with pytest.raises(CustodyPreparationRequired):
        provision(world)
    return world


def test_preparation_reuse_requires_identical_unsigned_content(prepared):
    world = prepared
    with pytest.raises(CustodyPreparationRequired):
        provision(world)
    files = list((world.state_dir / "node-key-custody/preparations").glob("*.json"))
    assert len(files) == 1
    files[0].write_bytes(files[0].read_bytes() + b"\n")
    with pytest.raises(
        CustodyReconciliationRequired, match="preparation content changed"
    ):
        provision(world)
    assert world.helper_calls == 0


def test_missing_signed_completion_is_drift_not_a_successful_probe(prepared):
    world = prepared
    world.authorize()
    with pytest.raises(BootstrapMutationRequired, match="signed custody completion"):
        provision(world, probe=True)
    assert world.helper_calls == 0


@pytest.mark.parametrize("failure", ["command", "no-receipt", "changed-binding"])
def test_helper_failure_cannot_be_recorded_as_custody_success(
    prepared, monkeypatch, failure
):
    world = prepared
    chain = world.authorize()
    original = world.run

    def failed(arguments, **kwargs):
        if str(arguments[0]).endswith("provision-node-action-keys.sh"):
            if failure == "command":
                raise BootstrapError("owned fake helper failed")
            if failure == "no-receipt":
                return "not a signed receipt"
            result = original(arguments, **kwargs)
            world.namespaces["gpu", world.context().namespace] = "changed-namespace"
            return result
        return original(arguments, **kwargs)

    monkeypatch.setattr(world, "run", failed)
    with pytest.raises(CustodyReconciliationRequired):
        provision(world)
    assert chain.exists() == (failure == "changed-binding")


def test_other_valid_authorization_cannot_replace_the_enrolled_receipt(prepared):
    world = prepared
    path = world.authorize()
    provision(world)
    chain = parse(Chain, path.read_bytes())
    head = chain.transactions[-1]
    authorization = head.authorization.statement.model_copy(
        update={"transaction_id": "0" * 64}
    )
    started = head.started.statement.model_copy(
        update={"authorization_sha256": statement_sha256(authorization)}
    )
    completed = head.completed.statement.model_copy(
        update={"started_sha256": statement_sha256(started)}
    )
    other = head.model_copy(
        update={
            "authorization": world.authorities.envelope(authorization, "approval"),
            "started": world.authorities.envelope(started, "provisioner"),
            "completed": world.authorities.envelope(completed, "provisioner"),
        }
    )
    path.write_bytes(canonical(Chain(transactions=[other])))
    with pytest.raises(CustodyReconciliationRequired, match="another authorization"):
        provision(world, probe=True)
    assert world.helper_calls == 1


def test_completed_resume_rechecks_registered_bytes_after_live_reads(
    prepared, monkeypatch
):
    world = prepared
    world.authorize()
    provision(world)
    original = world.run
    changed = False

    def change_input(arguments, **kwargs):
        nonlocal changed
        result = original(arguments, **kwargs)
        if not changed and "nodes" in arguments:
            changed = True
            path = world.root / "request.json"
            path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(world, "run", change_input)
    with pytest.raises(CustodyReconciliationRequired, match="inputs changed"):
        provision(world, probe=True)
    assert world.helper_calls == 1


def test_service_refuses_conflicting_custody_and_provisioning_scopes(prepared):
    world = prepared
    with pytest.raises(BootstrapError, match="context differs"):
        bootstrap_services.provision_node_action_keys(
            world,
            repository_root=world.repo,
            cpu_kubeconfig=world.cpu_kubeconfig,
            gpu_kubeconfig=world.gpu_kubeconfig,
            namespace=world.context().namespace,
            cluster=world.gpu,
            cluster_id="foreign-cluster",
            fleet_master_file=world.master_file,
            custody_context=world.context(),
        )
    assert world.helper_calls == 0


def test_profile_must_be_resolved_only_for_configured_custody(prepared):
    world = prepared
    with pytest.raises(CustodyPreparationRequired, match="known Runtime Profile"):
        bootstrap_custody_profile(world.state_dir, world.repo, {"spec": {}})
    assert (
        bootstrap_custody_profile(world.root / "unconfigured", world.repo, {"spec": {}})
        == "hyperpod-v1"
    )


def test_unselected_cluster_retains_legacy_provision_and_readonly_probe(prepared):
    world = prepared
    calls = []
    context = replace(
        world.context(),
        state_dir=world.root / "unconfigured",
        on_write=lambda: calls.append("write"),
    )

    def run(probe=False):
        return bootstrap_services.provision_node_action_keys(
            world,
            repository_root=world.repo,
            cpu_kubeconfig=world.cpu_kubeconfig,
            gpu_kubeconfig=world.gpu_kubeconfig,
            namespace=context.namespace,
            cluster=world.gpu,
            cluster_id=world.cluster_id,
            fleet_master_file=world.master_file,
            probe_only=probe,
            custody_context=context,
        )

    assert run() == {"cluster_id": world.cluster_id}
    assert calls == ["write"]
    assert run(probe=True) == {"cluster_id": world.cluster_id}
    assert calls == ["write"]
    assert world.helper_calls == 1
    assert not list((world.root / "receipts").glob("*.chain.json")), (
        "unselected cluster must not emit custody receipt chains"
    )
