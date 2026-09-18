from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from gpu_fault.admin.bootstrap_common import BootstrapMutationRequired, CommandRunner
from gpu_fault.admin.node_key_custody_admin import (
    CustodyPreparationRequired,
    CustodyReconciliationRequired,
)
from gpu_fault.admin.node_key_custody_admin_config import load_admin_custody
from gpu_fault.admin.node_key_custody_admin_entry import assert_site_custody_current
from gpu_fault.admin.node_key_custody_admin_probe import (
    CustodyReadRunner,
    current_binding,
    custody_verifier,
    live_node_uids,
    master_source,
    private_json,
)
from gpu_fault.admin.node_key_custody_crypto import parse
from gpu_fault.admin.node_key_custody_models import Authorization, CustodyError, Signed
from tests.admin._node_key_custody_admin_support import AdminWorld
from tests.admin.test_node_key_custody_admin import provision
from tests.deploy._node_action_key_api import encoded
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    world = AdminWorld(tmp_path, monkeypatch)
    world.configure()
    world.access()
    world.release_ready = True
    return world


class Reply(CommandRunner):
    def __init__(self, output):
        self.output = output
        self.calls = []

    def run(self, arguments, **kwargs):
        self.calls.append((list(arguments), kwargs))
        assert kwargs["capture"] and kwargs["sensitive"]
        assert not kwargs.get("mutate"), (
            "custody binding reads must not request mutation"
        )
        return self.output


@pytest.mark.parametrize("value", ["[]", "null", "invalid-json", "7"])
def test_private_binding_read_requires_an_object(value):
    with pytest.raises(CustodyError, match="invalid JSON"):
        private_json(Reply(value), ["kubectl", "get", "nodes"])


@pytest.mark.parametrize(
    "failure",
    [
        "empty",
        "pagination",
        "not-list",
        "duplicate-name",
        "duplicate-uid",
        "deleted",
        "foreign-label",
        "malformed",
    ],
)
def test_node_inventory_rejects_incomplete_or_ambiguous_bindings(prepared, failure):
    world = prepared
    document = copy.deepcopy(world.api.state["nodes"])
    if failure == "empty":
        document["items"] = []
    elif failure == "pagination":
        document["metadata"] = {"continue": "another-page"}
    elif failure == "not-list":
        document["items"] = {}
    elif failure.startswith("duplicate"):
        field = "name" if failure == "duplicate-name" else "uid"
        document["items"][1]["metadata"][field] = document["items"][0]["metadata"][
            field
        ]
    elif failure == "deleted":
        document["items"][0]["metadata"]["deletionTimestamp"] = "pending"
    elif failure == "foreign-label":
        document["items"][0]["metadata"]["labels"] = {}
    else:
        document["items"][0]["metadata"] = None
    with pytest.raises(CustodyError, match="incomplete or ambiguous"):
        live_node_uids(Reply(json.dumps(document)), world.context())


@pytest.mark.parametrize("failure", ["scope", "material", "missing", "deleted"])
def test_master_source_refuses_bad_identity_without_material_diagnostics(
    prepared, failure
):
    world = prepared
    document = json.loads(
        world.private_run(
            [
                *world.context().kubectl("cpu"),
                "get",
                "secret",
                "gpu-fault-node-installer",
            ]
        ).stdout
    )
    if failure == "scope":
        document["metadata"]["namespace"] = "other"
    elif failure == "material":
        document["data"]["node-action-secret"] = "!"
    elif failure == "missing":
        del document["metadata"]["uid"]
    else:
        document["metadata"]["deletionTimestamp"] = "pending"
    with pytest.raises(CustodyError, match="master source is invalid"):
        master_source(Reply(json.dumps(document)), world.context())


def resign_candidate(world):
    world.attestation.write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(
                    world.manifest_path.read_bytes()
                ).hexdigest()
            }
        )
    )
    (world.repo / "dist/current-attestation.bundle.json").write_bytes(
        world.release_key.sign(
            world.attestation.read_bytes(), ec.ECDSA(hashes.SHA256())
        )
    )


@pytest.mark.parametrize("failure", ["schema", "component", "foreign", "authority"])
def test_current_release_requires_complete_signed_independent_identity(
    prepared, failure
):
    world = prepared
    context = world.context()
    if failure == "schema":
        world.manifest["schema_version"] = 3
    elif failure == "component":
        del world.manifest["components"]["node_runtime"]["module_digest"]
    elif failure == "foreign":
        other = world.root / "foreign-release.json"
        other.write_text("{}")
        context = replace(context, release_manifest=other)
    else:
        world.release_key = world.authorities.keys["approval"]
        (world.state_dir / "release-signing/cosign.pub").write_bytes(
            world.release_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
    world.manifest_path.write_text(json.dumps(world.manifest))
    resign_candidate(world)
    with pytest.raises(CustodyError):
        current_binding(world, context, load_admin_custody(world.state_dir))
    assert world.api.state["calls"] == []
    assert world.helper_calls == 0


@pytest.mark.parametrize("location", ["request", "canonical"])
def test_release_identity_cannot_change_during_verification(
    prepared, monkeypatch, location
):
    world = prepared
    other = world.root / "same-release.json"
    other.write_bytes(world.manifest_path.read_bytes())
    context = replace(world.context(), release_manifest=other)
    original = world.run

    def changed(arguments, **kwargs):
        result = original(arguments, **kwargs)
        if any(str(arg).endswith("verify-release-attestation.py") for arg in arguments):
            path = other if location == "request" else world.manifest_path
            path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(world, "run", changed)
    with pytest.raises(CustodyError, match="changed during verification"):
        current_binding(world, context, load_admin_custody(world.state_dir))


@pytest.mark.parametrize(
    "change", ["missing", "uid", "pending", "extra-gpu-key", "cpu-key"]
)
def test_signed_completion_is_rechecked_against_actual_key_sources(prepared, change):
    world = prepared
    with pytest.raises(CustodyPreparationRequired):
        provision(world)
    world.authorize()
    provision(world)
    gpu = world.api.state["secrets"]["gpu"]
    if change == "missing":
        world.api.state["secrets"]["gpu"] = None
    elif change == "uid":
        gpu["metadata"]["uid"] = "different-secret"
    elif change == "pending":
        gpu["metadata"]["annotations"] = {"gpu-fault.io/node-action-key-rotation": "1"}
    elif change == "extra-gpu-key":
        gpu["data"]["extra-node"] = encoded("extra")
    else:
        world.api.state["secrets"]["cpu"]["data"]["node-a"] = encoded("different")
    with pytest.raises(CustodyReconciliationRequired, match="deployed key sources"):
        provision(world, probe=True)
    assert world.helper_calls == 1


def test_readonly_verifier_cannot_be_used_as_a_signer(prepared):
    world = prepared
    with pytest.raises(CustodyPreparationRequired):
        provision(world)
    world.authorize()
    verifier = custody_verifier(world, load_admin_custody(world.state_dir))
    approved = parse(Signed[Authorization], (world.root / "approved.json").read_bytes())
    with pytest.raises(CustodyError, match="signature command could not complete"):
        verifier.sign(approved.statement, "provisioner")
    assert not any(call[:3] == ["aws", "kms", "sign"] for call in world.calls), (
        "read-only custody verifier must not invoke KMS signing"
    )


def test_custody_reader_does_not_inherit_raw_environment_or_allow_mutation(monkeypatch):
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_SECRET", "synthetic-forbidden")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-forbidden")
    delegate = Reply("{}")
    reader = CustodyReadRunner(delegate)
    assert reader.run(["kubectl", "get", "nodes"], env={"unsafe": "value"}) == "{}"
    environment = delegate.calls[-1][1]["env"]
    assert not {"GPU_FAULT_NODE_ACTION_SECRET", "AWS_SECRET_ACCESS_KEY", "unsafe"} & (
        environment.keys()
    )
    with pytest.raises(BootstrapMutationRequired):
        reader.run(["kubectl", "create", "secret"], mutate=True)
    assert len(delegate.calls) == 1


def test_unenrolled_members_and_unconfigured_sites_do_not_acquire_custody_reads(
    tmp_path, monkeypatch
):
    world = AdminWorld(tmp_path, monkeypatch)
    assert_site_custody_current(world.site(), world)
    world.requests = {world.gpu.eks_arn + "-future": None}
    world.configure()
    assert_site_custody_current(world.site(), world)
    assert world.calls == []
