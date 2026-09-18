from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from gpu_fault.admin.node_key_custody import ProvisionCustody
from gpu_fault.admin.node_key_custody_chain import verify_chain
from gpu_fault.admin.node_key_custody_crypto import parse
from gpu_fault.admin.node_key_custody_models import Chain, CustodyError, canonical
from gpu_fault.node_action_keys import derive_node_action_secret
from tests.deploy._node_action_key_api import encoded, secret
from tests.deploy._node_key_custody_support import MASTER, ProvisionFixture
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def fixture(tmp_path):
    return ProvisionFixture(tmp_path)


def test_real_provisioning_captures_signed_prospective_custody_and_rotation(fixture):
    initial = fixture.session()
    installed = fixture.provision(initial)
    assert verify_chain(installed, fixture.crypto).activated is None
    assert fixture.events == ["write:gpu", "write:cpu"]
    active = fixture.activate(installed)
    old = dict(fixture.api.state["secrets"]["gpu"]["data"])
    rotation = fixture.session(active)
    rotated = fixture.provision(rotation)
    head = verify_chain(rotated, fixture.crypto)
    keys = head.completed.statement.gpu.keys
    assert keys["node-a"].generation == 2 and keys["node-b"].generation == 1
    assert (
        keys["node-b"]
        == installed.transactions[-1].completed.statement.gpu.keys["node-b"]
    )
    assert head.completed.statement.runtime_activation_proved is False
    assert Path(rotation.inputs.retired_key_file).read_bytes() == base64.b64decode(
        old["node-a"]
    )
    text = canonical(rotated).decode()
    assert MASTER.decode() not in text and base64.b64encode(MASTER).decode() not in text
    for data in (old, fixture.api.state["secrets"]["gpu"]["data"]):
        assert all(
            value not in text and base64.b64decode(value).decode() not in text
            for value in data.values()
        ), "custody receipts must not contain raw or encoded node keys"
    assert not any("exec" in call for call in fixture.api.state["calls"]), (
        "provisioning must not invoke kubectl exec"
    )


@pytest.mark.parametrize("plane", ["gpu", "cpu"])
def test_current_snapshot_cannot_be_promoted_to_installation_provenance(fixture, plane):
    matching = base64.b64encode(
        derive_node_action_secret(MASTER.decode(), "cluster-a", "node-a").encode()
    ).decode()
    fixture.api.state["secrets"][plane] = secret(plane, {"node-a": matching})
    session = fixture.session()
    with pytest.raises(CustodyError, match="pre-existing"):
        fixture.provision(session)
    assert fixture.events == []
    assert not list(session.directory.iterdir()), (
        "no fabricated start or completed receipt"
    )


def test_receipt_without_prior_real_activation_cannot_authorize_rotation(fixture):
    installed = fixture.provision(fixture.session())
    with pytest.raises(CustodyError, match="activation"):
        fixture.session(installed)


@pytest.mark.parametrize("field", ["producer_sha256", "witness_sha256"])
def test_producer_identity_is_independently_authorized(fixture, field):
    authorization = fixture.authorization().model_copy(update={field: "f" * 64})
    session = fixture.session(authorization=authorization)
    if field == "witness_sha256":
        # The producer cannot speak for the independent runtime witness.
        chain = fixture.provision(session)
        assert chain.transactions[-1].activated is None
    else:
        with pytest.raises(CustodyError, match="inputs differ"):
            fixture.provision(session)
        assert fixture.events == []


@pytest.mark.parametrize(
    "failure", ["wrong-master", "master-uid", "master-namespace", "node-uid", "anchor"]
)
def test_identity_mismatch_never_starts_key_writes(fixture, failure):
    session = fixture.session()
    if failure == "wrong-master":
        fixture.master_file.write_bytes(b"synthetic-other-master-" * 4)
    elif failure == "master-uid":
        fixture.master_document["metadata"]["uid"] = "recreated"
    elif failure == "master-namespace":
        fixture.master_document["metadata"]["namespace"] = "other"
    elif failure == "node-uid":
        fixture.api.state["nodes"]["items"][0]["metadata"]["uid"] = "recreated"
    base = fixture.run

    def runner(arguments, **kwargs):
        result = base(arguments, **kwargs)
        if failure == "anchor" and "namespace" in arguments:
            value = json.loads(result.stdout)
            value["metadata"]["uid"] = "foreign"
            result.stdout = json.dumps(value)
        return result

    with pytest.raises(CustodyError):
        fixture.provision(session, runner=runner)
    assert fixture.events == []


def test_signing_failure_prevents_transfer_and_partial_write_never_becomes_provenance(
    fixture, monkeypatch
):
    session = fixture.session()

    def refuse(*args, **kwargs):
        raise CustodyError("synthetic signer unavailable")

    monkeypatch.setattr(fixture.crypto, "sign", refuse)
    with pytest.raises(CustodyError, match="unavailable"):
        fixture.provision(session)
    assert fixture.events == []
    assert not list(session.directory.iterdir()), (
        "signing failure must not leave custody receipts"
    )


def test_failed_cpu_write_retains_only_start_and_cannot_reconstruct_a_completed_receipt(
    fixture,
):
    session = fixture.session()
    fixture.api.state["events"] = [{"on": "cpu:create", "returncode": 1}]
    with pytest.raises(CustodyError, match="direct acknowledgement"):
        fixture.provision(session)
    assert Path(str(session.prefix) + ".started.json").is_file(), (
        "failed CPU write must retain the prospective start receipt"
    )
    assert not Path(str(session.prefix) + ".chain.json").exists(), (
        "failed CPU write must not produce a completed custody chain"
    )
    with pytest.raises(CustodyError, match="already started"):
        ProvisionCustody(session.inputs, fixture.crypto, session.authorization, None)
    assert fixture.api.state["secrets"]["gpu"] is not None


def test_frozen_start_does_not_bless_concurrent_gpu_key_replacement(fixture):
    session = fixture.session()
    fixture.api.state["events"] = [
        {
            "on": "gpu:get-secret",
            "occurrence": 2,
            "merge_data": {"node-a": encoded("unapproved")},
        }
    ]
    with pytest.raises(CustodyError, match="plan changed"):
        fixture.provision(session)
    assert not Path(str(session.prefix) + ".chain.json").exists(), (
        "concurrent GPU key replacement must prevent custody completion"
    )
    assert fixture.api.state["secrets"]["cpu"] is None


@pytest.mark.parametrize("mode", [0o644, 0o400, 0o666])
def test_master_requires_owned_private_regular_file(fixture, mode):
    session = fixture.session()
    fixture.master_file.chmod(mode)
    with pytest.raises(CustodyError, match="controlled regular"):
        fixture.provision(session)
    assert fixture.events == []


def test_literal_master_in_environment_is_never_forwarded_to_any_writer(
    fixture, monkeypatch
):
    session = fixture.session()
    monkeypatch.setenv("SYNTHETIC_UNSAFE_VALUE", MASTER.decode())
    with pytest.raises(CustodyError, match="environment or argv"):
        fixture.provision(session)
    assert fixture.events == []


def test_request_loader_pins_authority_outside_the_request(fixture):
    session = fixture.session()
    request = fixture.root / "request.json"
    request.write_bytes(canonical(session.inputs))
    loaded = ProvisionCustody.load(
        request,
        fixture.authorities.trust_pin,
        crypto_factory=lambda path, pin: fixture.crypto,
    )
    chain = fixture.provision(loaded)
    assert parse(Chain, canonical(chain)) == chain


def test_unrelated_cpu_keys_are_preserved_but_no_unproven_scope_is_adopted(fixture):
    fixture.api.state["secrets"]["cpu"] = secret(
        "cpu", {"node-elsewhere": encoded("other-cluster")}
    )
    chain = fixture.provision(fixture.session())
    assert "node-elsewhere" in fixture.api.state["secrets"]["cpu"]["data"]
    assert set(chain.transactions[-1].completed.statement.cpu.keys) == {
        "node-a",
        "node-b",
    }


@pytest.mark.parametrize("plane", ["gpu", "cpu"])
def test_custody_create_ack_loss_cannot_be_reconstructed_from_current_membership(
    fixture, plane
):
    session = fixture.session()
    fixture.api.state["events"] = [
        {"on": plane + ":create", "returncode": 1, "lost_ack": True}
    ]
    with pytest.raises(CustodyError, match="direct acknowledgement"):
        fixture.provision(session)
    assert fixture.api.state["secrets"][plane] is not None
    assert Path(str(session.prefix) + ".started.json").is_file(), (
        "lost create acknowledgement must retain the prospective start receipt"
    )
    assert not Path(str(session.prefix) + ".chain.json").exists(), (
        "current Secret membership must not replace a lost create acknowledgement"
    )


def test_custody_does_not_adopt_a_concurrent_identical_gpu_secret_creator(fixture):
    session = fixture.session()
    original = fixture.run

    def race(arguments, **kwargs):
        if "create" in arguments and "gpu-context" in arguments:
            desired = json.loads(kwargs["input_text"])
            fixture.api.state["secrets"]["gpu"] = secret("gpu", desired["data"])
        return original(arguments, **kwargs)

    with pytest.raises(CustodyError, match="direct acknowledgement"):
        fixture.provision(session, runner=race)
    assert fixture.api.state["secrets"]["cpu"] is None
    assert not Path(str(session.prefix) + ".chain.json").exists(), (
        "another creator's identical GPU keys must not become custody provenance"
    )
