from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gpu_fault.admin import bootstrap, bootstrap_services
from gpu_fault.admin.bootstrap_common import (
    BootstrapMutationRequired,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.node_key_custody_admin import (
    CustodyPreparationRequired,
    CustodyReconciliationRequired,
)
from gpu_fault.admin.node_key_custody_admin_config import load_admin_custody
from gpu_fault.admin.node_key_custody_admin_entry import assert_site_custody_current
from gpu_fault.admin.node_key_custody_crypto import parse
from gpu_fault.admin.node_key_custody_models import Chain, CustodyError
from tests.admin._node_key_custody_admin_support import AdminWorld
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def world(tmp_path, monkeypatch):
    return AdminWorld(tmp_path, monkeypatch)


def provision(world, *, probe=False):
    return bootstrap_services.provision_node_action_keys(
        ReadOnlyProbeRunner(world) if probe else world,
        repository_root=world.repo,
        cpu_kubeconfig=world.cpu_kubeconfig,
        gpu_kubeconfig=world.gpu_kubeconfig,
        namespace=world.context().namespace,
        cluster=world.gpu,
        cluster_id=world.cluster_id,
        fleet_master_file=world.master_file,
        probe_only=probe,
        custody_context=world.context(),
    )


def test_explicit_admin_configuration_prepare_authorize_provision_and_readonly_resume(
    world, monkeypatch
):
    world.configure()
    assert world.helper_calls == 0
    assert not world.master_file.exists(), (
        "configuration must not invent deployment identities"
    )
    world.release_ready = True
    world.access()
    with pytest.raises(CustodyPreparationRequired, match="awaits independent"):
        provision(world)
    assert world.helper_calls == 0 and world.api.state["write_attempts"] == []
    chain_path = world.authorize()
    result = provision(world)
    assert result["runtime_activation"] == "NOT_PROVED"
    assert Path(result["custody_chain"]) == chain_path
    chain = parse(Chain, chain_path.read_bytes())
    binding = chain.transactions[0].authorization.statement.binding
    assert binding.nodes == {
        item["metadata"]["name"]: item["metadata"]["uid"]
        for item in world.api.state["nodes"]["items"]
    }
    assert (
        binding.site.gpu_namespace_uid
        == world.namespaces["gpu", world.context().namespace]
    )
    assert world.helper_calls == 1
    before = len(world.authorities.calls)
    assert provision(world, probe=True) == result
    assert provision(world) == result, (
        "missing bootstrap checkpoint must reuse signed completion"
    )
    assert world.helper_calls == 1
    assert not any(
        call[:3] == ["aws", "kms", "sign"] for call in world.authorities.calls[before:]
    ), "completed custody resume must not invoke KMS signing"
    monkeypatch.setattr(bootstrap, "discover_cluster", lambda *a, **k: world.gpu)
    assert_site_custody_current(world.site(), ReadOnlyProbeRunner(world))
    assert world.helper_calls == 1


def test_shape_only_probe_is_not_enough_when_custody_is_configured(world):
    world.configure()
    world.access()
    world.release_ready = True
    with pytest.raises(BootstrapMutationRequired):
        provision(world, probe=True)
    assert world.helper_calls == 0
    assert not (world.state_dir / "node-key-custody/preparations").exists(), (
        "read-only custody probe must not prepare new evidence"
    )


def test_known_release_is_required_before_collecting_namespace_bindings(world):
    world.configure()
    world.access()
    with pytest.raises(AssertionError, match="unfinished release"):
        provision(world)
    assert world.helper_calls == 0
    assert not world.api.state["calls"]


@pytest.mark.parametrize("change", ["namespace", "node", "master"])
def test_changed_preparation_identity_is_rejected_before_key_mutation(world, change):
    world.configure()
    world.access()
    world.release_ready = True
    with pytest.raises(CustodyPreparationRequired):
        provision(world)
    world.authorize()
    if change == "namespace":
        world.namespaces["gpu", world.context().namespace] = "changed"
    elif change == "node":
        world.api.state["nodes"]["items"][0]["metadata"]["uid"] = "changed"
    else:
        world.master_uid = "changed"
    with pytest.raises(CustodyReconciliationRequired, match="differs from current"):
        provision(world)
    assert world.helper_calls == 0


def test_partial_helper_write_preserves_intent_and_cannot_resume_from_matching_keys(
    world,
):
    world.configure()
    world.access()
    world.release_ready = True
    with pytest.raises(CustodyPreparationRequired):
        provision(world)
    chain = world.authorize()
    world.api.state["events"] = [
        {"on": "cpu:create", "returncode": 1, "lost_ack": True}
    ]
    with pytest.raises(CustodyReconciliationRequired):
        provision(world)
    assert (
        world.api.state["secrets"]["gpu"]["data"]
        == world.api.state["secrets"]["cpu"]["data"]
    )
    assert not chain.exists(), (
        "lost write acknowledgement must not produce a completed custody chain"
    )
    with pytest.raises(
        CustodyReconciliationRequired, match="incomplete custody intent"
    ):
        provision(world)
    assert world.helper_calls == 1


@pytest.mark.parametrize("which", ["request", "authorization", "trust"])
def test_enrolled_input_contents_cannot_change_silently(world, which):
    world.configure()
    world.access()
    world.release_ready = True
    with pytest.raises(CustodyPreparationRequired):
        provision(world)
    world.authorize()
    path = {
        "request": world.root / "request.json",
        "authorization": world.root / "approved.json",
        "trust": world.authorities.trust_path,
    }[which]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(CustodyReconciliationRequired):
        provision(world)
    assert world.helper_calls == 0


def test_corrupt_registration_never_disables_opt_in(world):
    world.configure()
    path = world.state_dir / "node-key-custody/registration.json"
    path.unlink()
    with pytest.raises(CustodyError):
        load_admin_custody(world.state_dir)


def test_explicit_completed_custody_preflight_does_not_create_missing_proof(
    world, monkeypatch
):
    world.configure()
    world.access()
    world.release_ready = True
    monkeypatch.setattr(bootstrap, "discover_cluster", lambda *a, **k: world.gpu)
    with pytest.raises(CustodyReconciliationRequired, match="public preflight"):
        assert_site_custody_current(world.site(), world)
    assert world.helper_calls == 0
    assert not any(
        call[:3] == ["aws", "kms", "sign"] for call in world.authorities.calls
    ), "public preflight must not invoke KMS signing"


def test_admin_to_helper_handoff_rejects_changed_authorization_before_key_write(world):
    world.configure()
    world.access()
    world.release_ready = True
    with pytest.raises(CustodyPreparationRequired):
        provision(world)
    world.authorize()

    def change():
        path = world.root / "approved.json"
        path.write_bytes(path.read_bytes() + b"\n")

    context = replace(world.context(), on_write=change)
    with pytest.raises(CustodyReconciliationRequired, match="pinned request"):
        bootstrap_services.provision_node_action_keys(
            world,
            repository_root=world.repo,
            cpu_kubeconfig=world.cpu_kubeconfig,
            gpu_kubeconfig=world.gpu_kubeconfig,
            namespace=context.namespace,
            cluster=world.gpu,
            cluster_id=world.cluster_id,
            fleet_master_file=world.master_file,
            custody_context=context,
        )
    assert world.api.state["write_attempts"] == []
    assert not list((world.root / "receipts").glob("*.started.json")), (
        "changed authorization must not issue a signed start receipt"
    )
