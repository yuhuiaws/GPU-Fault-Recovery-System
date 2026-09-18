from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.admin.node_key_custody import (
    ProvisionCustody,
    key_digests,
    namespace_uid,
    provisioning_input_identity,
)
from gpu_fault.admin.node_key_custody_chain import authorize_now
from gpu_fault.admin.node_key_custody_models import CustodyError, canonical
from tests.deploy._node_action_key_api import encoded
from tests.deploy._node_key_custody_support import MASTER, ProvisionFixture
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def fixture(tmp_path):
    return ProvisionFixture(tmp_path)


def test_pinned_loader_rechecks_content_after_verifier_construction(fixture):
    session = fixture.session()
    path = fixture.root / "pinned-request.json"
    path.write_bytes(canonical(session.inputs))
    identity = provisioning_input_identity(path, fixture.authorities.trust_pin)

    def crypto_factory(*args):
        path.write_bytes(path.read_bytes() + b"\n")
        return fixture.authorities.crypto()

    with pytest.raises(CustodyError, match="inputs changed while loading"):
        ProvisionCustody.load(
            path,
            fixture.authorities.trust_pin,
            expected_input_sha256=identity,
            crypto_factory=crypto_factory,
        )
    assert not list(session.directory.iterdir()), (
        "changed pinned inputs must not create custody receipts"
    )
    assert fixture.api.state["write_attempts"] == []


def state(fixture, plane):
    document = copy.deepcopy(fixture.api.state["secrets"][plane])
    return SimpleNamespace(
        document=document,
        data=document["data"],
        uid=document["metadata"]["uid"],
        version=document["metadata"]["resourceVersion"],
    )


@pytest.mark.parametrize(
    "value", ["?", "eA==", base64.b64encode(b" " + b"k" * 40).decode(), 7]
)
def test_key_digest_errors_are_generic_and_never_return_partial_maps(value):
    with pytest.raises(CustodyError, match="invalid or whitespace"):
        key_digests({"node-a": encoded("valid"), "node-b": value})


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"metadata": {}},
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": "wrong", "uid": "uid"},
        },
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": "wanted", "uid": ""},
        },
    ],
)
def test_namespace_anchor_requires_complete_unambiguous_identity(document):
    with pytest.raises(CustodyError, match="anchor is invalid"):
        namespace_uid(document, "wanted")


@pytest.mark.parametrize("failure", ["missing-previous", "wrong-previous", "manifest"])
def test_authorized_request_still_requires_its_actual_predecessor_and_manifest(
    fixture, failure
):
    if failure == "manifest":
        authorization = fixture.authorization()
        fixture.manifest.write_text('{"synthetic":"different"}')
        previous = None
    else:
        previous = fixture.activate(fixture.provision(fixture.session()))
        authorization = fixture.authorization(previous)
        if failure == "missing-previous":
            previous = None
        else:
            authorization = authorization.model_copy(
                update={"previous_receipt_sha256": "f" * 64}
            )
    with pytest.raises(CustodyError):
        fixture.session(previous, authorization=authorization)


@pytest.mark.parametrize("location", ["missing", "receipt", "artifacts"])
def test_retired_key_can_only_be_retained_in_a_separate_controlled_directory(
    fixture, location
):
    previous = fixture.activate(fixture.provision(fixture.session()))
    session = fixture.session(previous)
    if location == "missing":
        path = None
    elif location == "receipt":
        path = str(session.directory / "retired")
    else:
        directory = fixture.root / "artifacts"
        directory.mkdir(mode=0o700)
        path = str(directory / "retired")
    inputs = session.inputs.model_copy(update={"retired_key_file": path})
    with pytest.raises(CustodyError):
        ProvisionCustody(inputs, fixture.crypto, session.authorization, previous)


def test_start_transfer_and_completion_cannot_precede_master_and_signed_intent(fixture):
    session = fixture.session()
    data = {"node-a": encoded("a"), "node-b": encoded("b")}
    with pytest.raises(CustodyError, match="master source"):
        session.begin(None, None, data)
    with pytest.raises(CustodyError, match="signed start"):
        session.transfer({"data": data}, gpu=True)
    with pytest.raises(CustodyError, match="signed start"):
        session.finish(SimpleNamespace(), SimpleNamespace())
    assert not list(session.directory.iterdir()), (
        "custody receipts require a verified master source and signed start"
    )


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_post_start_transfer_and_final_readback_must_match_the_signed_plan(
    fixture, plane
):
    session = fixture.session()
    fixture.provision(session)
    gpu, cpu = state(fixture, "gpu"), state(fixture, "cpu")
    changed = gpu if plane == "gpu" else cpu
    changed.data["node-a"] = encoded("unapproved")
    with pytest.raises(CustodyError, match="signed plan"):
        session.transfer(changed.document, gpu=plane == "gpu")
    with pytest.raises(CustodyError, match="final readback"):
        session.finish(gpu, cpu)


@pytest.mark.parametrize("leak", ["master", "node", "encoded-master"])
def test_metadata_cannot_smuggle_material_alongside_an_approved_key_map(fixture, leak):
    session = fixture.session()
    fixture.provision(session)
    document = copy.deepcopy(fixture.api.state["secrets"]["gpu"])
    value = {
        "master": MASTER.decode(),
        "encoded-master": base64.b64encode(MASTER).decode(),
        "node": base64.b64decode(document["data"]["node-a"]).decode(),
    }[leak]
    document["metadata"]["annotations"] = {"unapproved": "prefix:" + value}
    with pytest.raises(CustodyError, match="metadata"):
        session.transfer(document, gpu=True)


@pytest.mark.parametrize("bad", ["master", "duplicate"])
def test_node_key_plan_cannot_alias_the_master_or_a_peer(fixture, bad):
    session = fixture.session()
    session.master(fixture.master_document, MASTER)
    first = base64.b64encode(MASTER).decode() if bad == "master" else encoded("shared")
    second = encoded("other") if bad == "master" else first
    with pytest.raises(CustodyError, match="distinct node-scoped"):
        session.begin(None, None, {"node-a": first, "node-b": second})
    assert not list(session.directory.iterdir()), (
        "aliased node keys must not receive a signed start receipt"
    )


@pytest.mark.parametrize(
    "failure", ["gpu-identity", "cpu-bytes", "no-change", "peer-change"]
)
def test_rotation_checks_predecessor_and_changed_set_before_issuing_start(
    fixture, failure
):
    previous = fixture.activate(fixture.provision(fixture.session()))
    session = fixture.session(previous)
    session.master(fixture.master_document, MASTER)
    gpu, cpu = state(fixture, "gpu"), state(fixture, "cpu")
    desired = dict(gpu.data)
    if failure == "gpu-identity":
        gpu.uid = "foreign"
    elif failure == "cpu-bytes":
        cpu.data["node-a"] = encoded("foreign")
    elif failure == "peer-change":
        desired["node-b"] = encoded("unapproved-peer-change")
    with pytest.raises(CustodyError):
        session.begin(gpu, cpu, desired)
    assert not list(session.directory.iterdir()), (
        "invalid rotation predecessor or key changes must not issue receipts"
    )


def test_metadata_cas_retry_reuses_the_same_signed_intent_and_rotation_material(
    fixture,
):
    previous = fixture.activate(fixture.provision(fixture.session()))
    session = fixture.session(previous)
    fixture.api.state["events"] = [
        {
            "on": "gpu:replace",
            "occurrence": 1,
            "returncode": 1,
            "merge_metadata": {"labels": {"unrelated-owner": "kept"}},
        }
    ]
    chain = fixture.provision(session)
    assert len(chain.transactions) == 2
    assert len(list(session.directory.glob("*.started.json"))) == 1
    assert fixture.api.state["secrets"]["gpu"]["metadata"]["labels"] == {
        "unrelated-owner": "kept"
    }


def test_rotation_request_loader_verifies_the_actual_prior_signed_chain(fixture):
    previous = fixture.activate(fixture.provision(fixture.session()))
    session = fixture.session(previous)
    path = fixture.root / "request.json"
    path.write_bytes(canonical(session.inputs))
    loaded = ProvisionCustody.load(
        path, fixture.authorities.trust_pin, crypto_factory=lambda *args: fixture.crypto
    )
    assert loaded.previous == previous
    assert loaded.predecessor == previous.transactions[-1]


def test_expired_authorization_cannot_be_used_at_a_later_write_boundary(
    fixture, monkeypatch
):
    session = fixture.session()
    fixture.provision(session)
    after = session.statement.expires_at + timedelta(seconds=1)
    with pytest.raises(CustodyError, match="approved window"):
        authorize_now(session.statement, after)
    from gpu_fault.admin import node_key_custody as module

    monkeypatch.setattr(module, "datetime", SimpleNamespace(now=lambda _tz: after))
    with pytest.raises(CustodyError, match="approved window"):
        session.transfer(fixture.api.state["secrets"]["gpu"], gpu=True)
    assert datetime.now(timezone.utc) < after


def test_authorized_identity_fields_cannot_leak_master_material_into_receipts(fixture):
    fixture.binding = fixture.binding.model_copy(
        update={
            "site": fixture.binding.site.model_copy(
                update={"site_name": MASTER.decode()}
            )
        }
    )
    session = fixture.session()
    with pytest.raises(CustodyError, match="identity contains credential"):
        fixture.provision(session)
    assert fixture.events == []
    assert not list(session.directory.iterdir()), (
        "identity fields containing master material must not reach receipts"
    )


def test_rotation_receipt_id_cannot_be_the_retired_raw_key(fixture):
    previous = fixture.activate(fixture.provision(fixture.session()))
    old = base64.b64decode(
        fixture.api.state["secrets"]["gpu"]["data"]["node-a"]
    ).decode()
    authorization = fixture.authorization(previous).model_copy(
        update={"transaction_id": old}
    )
    session = fixture.session(previous, authorization=authorization)
    before = list(fixture.events)
    with pytest.raises(CustodyError, match="identity contains credential"):
        fixture.provision(session)
    assert fixture.events == before
    assert not list(session.directory.iterdir()), (
        "a receipt identifier containing a retired key must not be persisted"
    )
