from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from gpu_fault.admin.node_key_custody_models import (
    Chain,
    CustodyError,
    statement_sha256,
)
from scripts.e2e.regional import auth015_custody as custody
from scripts.e2e.regional import auth015_custody_inputs as inputs
from scripts.e2e.regional import identity_acceptance_auth as auth
from scripts.e2e.regional.auth015_deployed import capture_snapshot
from scripts.e2e.regional.auth015_release import verify_release_inputs
from tests.regional._auth015_custody_support import RuntimeFixture
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    fixture = RuntimeFixture(tmp_path, monkeypatch)
    fixture.install_runtime_keys()
    try:
        yield fixture
    finally:
        fixture.close()


def test_external_trust_pin_and_verification_readback_are_required(
    runtime, monkeypatch
):
    values = runtime.inputs()
    with pytest.raises(CustodyError, match="external pin"):
        inputs.custody_input_identity(replace(values, trust_sha256="f" * 64))
    factory = inputs.CustodyCrypto

    def changed(path, pin):
        result = factory(path, pin)
        values.descriptor_path.write_bytes(values.descriptor_path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(inputs, "CustodyCrypto", changed)
    with pytest.raises(CustodyError, match="during verification"):
        inputs.verify_custody_inputs(values)


def test_initial_activation_cannot_claim_retired_key_evidence(runtime):
    runtime.retired_path = runtime.root / "private-old"
    runtime.retired_path.write_bytes(b"synthetic-private-old-" * 4)
    runtime.retired_path.chmod(0o600)
    with pytest.raises(CustodyError, match="must not supply"):
        inputs.retired_key(runtime.inputs(), runtime.chain)


def test_rotation_requires_its_old_key_file_and_valid_encoding(runtime):
    runtime.witness()
    runtime.rotate()
    values = runtime.inputs()
    descriptor = json.loads(values.descriptor_path.read_text())
    descriptor["retired_key_file"] = None
    values.descriptor_path.write_text(json.dumps(descriptor))
    with pytest.raises(CustodyError, match="controlled retired key"):
        inputs.retired_key(values, runtime.chain)
    invalid = b"\xff" * 48
    runtime.retired_path.write_bytes(invalid)
    previous, current = runtime.chain.transactions
    keys = dict(previous.completed.statement.gpu.keys)
    keys["node-a"] = keys["node-a"].model_copy(
        update={"sha256": hashlib.sha256(invalid).hexdigest()}
    )
    previous = previous.model_copy(
        update={
            "completed": previous.completed.model_copy(
                update={
                    "statement": previous.completed.statement.model_copy(
                        update={
                            "gpu": previous.completed.statement.gpu.model_copy(
                                update={"keys": keys}
                            )
                        }
                    )
                }
            )
        }
    )
    with pytest.raises(CustodyError, match="encoding"):
        inputs.retired_key(runtime.inputs(), Chain(transactions=[previous, current]))


@pytest.mark.parametrize("failure", ["cluster", "duplicate-node", "deleting-node"])
def test_live_inventory_cannot_change_cluster_or_hide_a_node(
    runtime, monkeypatch, failure
):
    if failure == "cluster":
        original = runtime.site.gpu

        def changed(target, *arguments, **kwargs):
            value = original(target, *arguments, **kwargs)
            if arguments[:2] == ("get", "namespace"):
                document = json.loads(value)
                document["metadata"]["uid"] = "different"
                return json.dumps(document)
            return value

        monkeypatch.setattr(runtime.site, "gpu", changed)
    elif failure == "duplicate-node":
        runtime.site.raw["nodes"]["items"].append(runtime.site.raw["nodes"]["items"][0])
    else:
        runtime.site.raw["nodes"]["items"][0]["metadata"]["deletionTimestamp"] = (
            "2026-01-01T00:00:00Z"
        )
    with pytest.raises(CustodyError):
        custody.live_anchors(
            runtime.site, runtime.site.target, runtime.provisioning.binding
        )


@pytest.mark.parametrize("failure", ["selection", "release", "heartbeat"])
def test_runtime_binding_checks_authorized_selection_release_and_post_provision_heartbeat(
    runtime, failure
):
    release = verify_release_inputs(runtime.site.files.inputs())
    snapshot = capture_snapshot(
        runtime.site, runtime.site.target, ("node-a", "node-b"), release
    )
    nodes = ("node-a", "node-b")
    if failure == "selection":
        nodes = ("node-a", "node-a")
    elif failure == "release":
        release = replace(release, node_wheel_sha256="f" * 64)
    else:
        snapshot.agents["node-a"] = snapshot.agents["node-a"].model_copy(
            update={
                "last_seen_at": runtime.chain.transactions[
                    -1
                ].completed.statement.observed_at
            }
        )
    with pytest.raises(CustodyError):
        custody.bind_runtime(snapshot, release, runtime.chain, nodes)


def test_a_valid_chain_cannot_authorize_different_witness_source(runtime):
    head = runtime.chain.transactions[-1]
    authorities = runtime.provisioning.authorities
    authorization = head.authorization.statement.model_copy(
        update={"witness_sha256": "f" * 64}
    )
    start = head.started.statement.model_copy(
        update={"authorization_sha256": statement_sha256(authorization)}
    )
    complete = head.completed.statement.model_copy(
        update={"started_sha256": statement_sha256(start)}
    )
    runtime.chain = Chain(
        transactions=[
            head.model_copy(
                update={
                    "authorization": authorities.envelope(authorization, "approval"),
                    "started": authorities.envelope(start, "provisioner"),
                    "completed": authorities.envelope(complete, "provisioner"),
                }
            )
        ]
    )
    with pytest.raises(auth.IdentityCaseFailure, match="authorized code"):
        runtime.witness()


def test_release_signer_cannot_double_as_the_custody_witness(runtime):
    files = runtime.site.files
    files.public_key_path.write_bytes((runtime.root / "witness.pem").read_bytes())
    files.descriptor["cosign_public_key_sha256"] = hashlib.sha256(
        files.public_key_path.read_bytes()
    ).hexdigest()
    files.write_descriptor()
    with pytest.raises(auth.IdentityCaseFailure, match="independent of the release"):
        runtime.witness()


@pytest.mark.parametrize(
    "failure", ["verdict", "identity", "activation-flag", "secret-readback", "inputs"]
)
def test_partial_protocol_or_readback_drift_cannot_be_signed_as_activation(
    runtime, monkeypatch, failure
):
    original = custody.prove_deployed_protocol

    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        if failure == "verdict":
            result["verdict"] = "FAIL"
        elif failure == "identity":
            result["identity"] = {}
        elif failure == "activation-flag":
            result["supplied_retired_key_denied"] = True
        elif failure == "secret-readback":
            runtime.provisioning.api.state["secrets"]["cpu"]["metadata"][
                "resourceVersion"
            ] = "new-version"
        else:
            path = runtime.root / "custody-proof.json"
            path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(custody, "prove_deployed_protocol", change)
    with pytest.raises(auth.IdentityCaseFailure):
        runtime.witness()
    assert runtime.chain.transactions[-1].activated is None
    assert not list(
        (runtime.root / "witness-1").glob("auth015-custody-chain-*.json")
    ), "partial protocol or readback drift must not produce a signed activation chain"


def test_private_transport_exception_is_reported_only_by_type(runtime, monkeypatch):
    sentinel = "synthetic-private-response-" * 8

    def refuse(*args, **kwargs):
        raise ValueError(sentinel)

    monkeypatch.setattr(custody, "live_keys", refuse)
    with pytest.raises(auth.IdentityCaseFailure) as caught:
        runtime.witness()
    assert sentinel not in str(caught.value)
    assert caught.value.details["failure_stage"] == "live-identity"
