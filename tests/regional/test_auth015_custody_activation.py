from __future__ import annotations

import base64
import json

import pytest

from gpu_fault.admin.node_key_custody_chain import verify_chain
from gpu_fault.node_agent.ledger import NodeActionLedger
from scripts.e2e.regional import auth015_custody as custody
from scripts.e2e.regional import auth015_protocol as protocol
from scripts.e2e.regional import identity_acceptance_auth as auth
from tests.deploy._node_key_custody_support import MASTER
from tests.regional._auth015_custody_support import RuntimeFixture
from tests.regional._cov95_auth015_support import KEY_A, KEY_B, AgentPair
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    fixture = RuntimeFixture(tmp_path, monkeypatch)
    try:
        yield fixture
    finally:
        fixture.close()


def test_prospective_install_rotate_and_independent_runtime_witness_chain(runtime):
    runtime.install_runtime_keys()
    initial = runtime.witness()
    assert initial["verdict"] == "FAIL", "installation alone is not a rotation proof"
    assert initial["checks"]["installation_time_master_custody"] is True
    assert initial["checks"]["deployed_node_a_key_activation"] is False
    assert set(initial["not_evaluated"]) == {"deployed_node_a_key_activation"}
    old_key = runtime.site.pair.executors["node-a"].secret
    peer_key = runtime.site.pair.executors["node-b"].secret
    runtime.rotate()
    writes = list(runtime.provisioning.events)
    runtime.install_runtime_keys()
    result = runtime.witness()
    assert result["verdict"] == "PASS"
    assert result["installation_custody_proved"] is True
    assert result["rotated_key_activation_proved"] is True
    assert result["deployed_protocol"]["rotated_key_activation_proved"] is False
    assert result["not_evaluated"] == {}
    assert all(result["checks"].values()), (
        "successful rotation witness must satisfy every custody check"
    )
    assert runtime.provisioning.events == writes, (
        "witness must not provision or restore keys"
    )
    assert runtime.site.pair.executors["node-a"].secret != old_key
    assert runtime.site.pair.executors["node-b"].secret == peer_key
    head = verify_chain(
        runtime.chain, runtime.provisioning.crypto, require_activation=True
    )
    assert head.activated.statement.retired_key_denied is True
    assert head.activated.statement.protocol_sha256 == custody.json_sha256(
        result["protocol_evidence"]
    )
    assert head.activated.statement.runtime_identity_sha256 == custody.json_sha256(
        result["deployed_protocol"]["identity"]
    )
    observations = result["activation_protocol"]["protocol"]["observations"]
    assert {item["check"] for item in observations if item["status"] == 401} == {
        "sibling_command_signature_rejected",
        "sibling_result_signature_rejected",
        "retired_command_signature_rejected",
        "retired_result_signature_rejected",
    }
    assert {
        item["check"]
        for item in result["deployed_protocol"]["protocol"]["observations"]
        if item["status"] == 401
    } == {"sibling_command_signature_rejected", "sibling_result_signature_rejected"}
    serialized = json.dumps(result)
    for value in (
        MASTER.decode(),
        old_key,
        peer_key,
        runtime.site.pair.executors["node-a"].secret,
    ):
        assert (
            value not in serialized
            and base64.b64encode(value.encode()).decode() not in serialized
        )
    assert all(
        "submit" not in request.full_url or b"FREEZE_EVIDENCE" in request.data
        for request in runtime.site.pair.requests
    ), "activation witness may submit only FREEZE_EVIDENCE commands"


def test_current_secret_membership_without_deployed_activation_cannot_pass(runtime):
    runtime.install_runtime_keys()
    runtime.witness()
    old = runtime.site.pair.executors["node-a"].secret
    runtime.rotate()
    signatures_before = [
        call
        for call in runtime.provisioning.authorities.calls
        if call[:3] == ["aws", "kms", "sign"]
    ]
    with pytest.raises(auth.IdentityCaseFailure, match="failed closed"):
        runtime.witness()
    assert runtime.site.pair.executors["node-a"].secret == old
    assert runtime.chain.transactions[-1].activated is None
    assert [
        call
        for call in runtime.provisioning.authorities.calls
        if call[:3] == ["aws", "kms", "sign"]
    ] == signatures_before


@pytest.mark.parametrize(
    "failure",
    [
        "retired",
        "cpu-key",
        "gpu-key",
        "cpu-uid",
        "gpu-uid",
        "namespace",
        "site",
        "node-uid",
        "release",
        "witness-source",
        "peer-incarnation",
    ],
)
def test_bound_chain_or_runtime_drift_never_receives_activation_receipt(
    runtime, failure
):
    runtime.install_runtime_keys()
    runtime.witness()
    runtime.rotate()
    runtime.install_runtime_keys()
    if failure == "retired":
        runtime.retired_path.write_bytes(b"synthetic-wrong-old-key-" * 4)
    elif failure in {"cpu-key", "gpu-key"}:
        plane = failure.split("-")[0]
        runtime.provisioning.api.state["secrets"][plane]["data"]["node-a"] = (
            base64.b64encode(b"synthetic-drift-" * 5).decode()
        )
    elif failure in {"cpu-uid", "gpu-uid"}:
        plane = failure.split("-")[0]
        runtime.provisioning.api.state["secrets"][plane]["metadata"]["uid"] = (
            "recreated"
        )
    elif failure == "namespace":
        runtime.site.namespace = "other"
    elif failure == "site":
        runtime.site.config["site_name"] = "other"
    elif failure == "node-uid":
        runtime.site.raw["nodes"]["items"][0]["metadata"]["uid"] = "recreated"
    elif failure == "release":
        runtime.site.raw["release_state"]["agent_config_digest"] = "9" * 64
    elif failure == "witness-source":
        head = runtime.chain.transactions[-1]
        altered = head.authorization.statement.model_copy(
            update={"witness_sha256": "f" * 64}
        )
        runtime.chain.transactions[-1] = head.model_copy(
            update={
                "authorization": runtime.provisioning.authorities.envelope(
                    altered, "approval"
                )
            }
        )
    else:
        runtime.site.raw["agent_snapshot"]["agents"]["node-b"][
            "agent_incarnation_id"
        ] = "restarted"
    with pytest.raises(auth.IdentityCaseFailure):
        runtime.witness()
    assert runtime.chain.transactions[-1].activated is None


def test_an_empty_provenance_input_does_not_read_a_master_or_touch_any_site(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("missing provenance must be rejected before I/O or a mutation")

    monkeypatch.setattr(auth, "prepare_auth015", forbidden)
    monkeypatch.setattr(auth, "run", forbidden)
    result = auth.run_auth015(
        object(),
        object(),
        nodes=("node-a", "node-b"),
        case_dir=tmp_path,
        fleet_master_file=tmp_path / "nonexistent-master",
        host_probe_image="unused",
        release_inputs=None,
    )
    assert result["verdict"] == "FAIL"
    assert result["requires_new_authorized_evidence"] is True
    assert len(result["not_evaluated"]) == 3
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("key", ["short", KEY_A, KEY_B, " " + "x" * 32])
def test_retired_challenge_requires_a_distinct_exact_old_key(
    tmp_path, monkeypatch, key
):
    pair = AgentPair(tmp_path, monkeypatch)
    try:
        with pytest.raises(protocol.Auth015ProofError, match="invalid"):
            protocol.prove_signatures(
                pair.targets["node-a"], pair.targets["node-b"], retired_key=key
            )
        assert pair.requests == []
    finally:
        pair.close()


@pytest.mark.parametrize("accepted", ["command", "result"])
def test_an_endpoint_that_accepts_the_old_key_is_refused(
    tmp_path, monkeypatch, accepted
):
    pair = AgentPair(tmp_path, monkeypatch)
    original = pair.bounded_request
    old = "synthetic-retired-key-" + "r" * 48

    def wrong_accept(url, **kwargs):
        status, body = original(url, **kwargs)
        # The two old-key requests follow both sibling-key denials.
        if len(pair.requests) == (6 if accepted == "command" else 7):
            return 404, b'{"detail":"node action command is unknown"}'
        return status, body

    monkeypatch.setattr(protocol, "bounded_request", wrong_accept)
    try:
        with pytest.raises(protocol.Auth015ProofError, match="retired_"):
            protocol.prove_signatures(
                pair.targets["node-a"], pair.targets["node-b"], retired_key=old
            )
    finally:
        pair.close()


def test_failed_local_precheck_never_invokes_the_runtime_witness(runtime, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a failed focused check must block the witness")

    monkeypatch.setattr(auth, "run_custody_acceptance", forbidden)
    with pytest.raises(auth.IdentityAcceptanceError, match="focused"):
        auth.run_auth015(
            runtime.site,
            runtime.site.target,
            nodes=("node-a", "node-b"),
            case_dir=runtime.root,
            release_inputs=runtime.site.files.inputs(),
            custody_inputs=runtime.inputs(),
            focused_tests={"passed": False},
        )


def test_owned_agent_pair_releases_both_ledgers_once(tmp_path, monkeypatch):
    closed = []
    original = NodeActionLedger.close

    def close(ledger):
        closed.append(ledger)
        original(ledger)

    monkeypatch.setattr(NodeActionLedger, "close", close)
    pair = AgentPair(tmp_path, monkeypatch)
    pair.close()
    assert len(closed) == 2
    pair.close()
    assert len(closed) == 2, "idempotent fixture teardown must not leak or double-close"


def test_supplied_wrong_key_denial_is_not_historical_rotation_proof(
    tmp_path, monkeypatch
):
    pair = AgentPair(tmp_path, monkeypatch)
    try:
        result = protocol.prove_signatures(
            pair.targets["node-a"],
            pair.targets["node-b"],
            retired_key="synthetic-never-deployed-" + "r" * 48,
        )
        assert result["supplied_retired_key_denied"] is True
        assert result["rotated_key_activation_proved"] is False
    finally:
        pair.close()
