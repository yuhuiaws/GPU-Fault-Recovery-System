from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.admin.node_key_custody import ProvisionInputs
from gpu_fault.admin.node_key_custody_admin import (
    CustodyPreparationRequired,
    CustodyReconciliationRequired,
)
from gpu_fault.admin.node_key_custody_admin_config import (
    load_admin_custody,
    registration_path,
)
from gpu_fault.admin.node_key_custody_crypto import parse
from gpu_fault.admin.node_key_custody_models import (
    Activated,
    Authorization,
    Chain,
    CustodyError,
    RuntimeNode,
    Signed,
    canonical,
    statement_sha256,
)
from tests.admin._node_key_custody_admin_support import AdminWorld
from tests.admin.test_node_key_custody_admin import provision
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def installed(tmp_path, monkeypatch):
    world = AdminWorld(tmp_path, monkeypatch)
    world.configure()
    world.access()
    world.release_ready = True
    with pytest.raises(CustodyPreparationRequired):
        provision(world)
    chain_path = world.authorize()
    provision(world)
    chain = parse(Chain, chain_path.read_bytes())
    head = chain.transactions[-1]
    binding = head.authorization.statement.binding
    activation = Activated(
        completed_sha256=statement_sha256(head.completed.statement),
        binding_sha256=statement_sha256(binding),
        observed_at=datetime.now(timezone.utc),
        nodes={
            name: RuntimeNode(
                node_uid=uid,
                boot_id="owned-fake-boot-" + name,
                agent_incarnation_id="owned-fake-agent-" + name,
                agent_generation=1,
                endpoint=f"https://{name}.invalid:9099",
                certificate_sha256="7" * 64,
            )
            for name, uid in binding.nodes.items()
        },
        runtime_identity_sha256="8" * 64,
        protocol_sha256="9" * 64,
        retired_key_denied=False,
    )
    # The independent witness is a synthetic authority at this administrator
    # boundary; AUTH-015's loopback tests exercise its actual protocol separately.
    activated = chain.model_copy(
        update={
            "transactions": [
                head.model_copy(
                    update={
                        "activated": world.authorities.envelope(activation, "witness")
                    }
                )
            ]
        }
    )
    previous_path = world.root / "previous-activated.json"
    previous_path.write_bytes(canonical(activated))
    private = world.root / "private-retired"
    private.mkdir(mode=0o700)
    now = datetime.now(timezone.utc)
    authorization = head.authorization.statement.model_copy(
        update={
            "transaction_id": "0" * 64,
            "purpose": "rotate",
            "rotate_node": "node-a",
            "previous_receipt_sha256": statement_sha256(activated.transactions[-1]),
            "not_before": now - timedelta(seconds=1),
            "expires_at": now + timedelta(minutes=15),
        }
    )
    authorized = world.root / "rotation-authorization.json"
    authorized.write_bytes(
        canonical(world.authorities.envelope(authorization, "approval"))
    )
    request = ProvisionInputs(
        trust=str(world.authorities.trust_path),
        authorization=str(authorized),
        release_manifest=str(world.manifest_path),
        previous_chain=str(previous_path),
        state_directory=str(chain_path.parent),
        retired_key_file=str(private / "retired-key"),
    )
    request_path = world.root / "rotation-request.json"
    request_path.write_bytes(canonical(request))
    world.requests[world.gpu.eks_arn] = str(request_path)
    from gpu_fault.admin import node_key_custody_activation as activation
    from tests.admin._security_activation_support import ActivationDouble

    world.activation_io = ActivationDouble(
        keys=lambda: world.api.state["secrets"]["gpu"]["data"],
        authorization=authorization,
    )
    monkeypatch.setattr(activation, "activation_io", lambda *_args: world.activation_io)
    return world, chain_path, request_path


def test_explicit_authorized_successor_rotates_through_the_real_helper_and_resumes(
    installed,
):
    world, _, request_path = installed
    old = dict(world.api.state["secrets"]["gpu"]["data"])
    world.configure()
    assert world.helper_calls == 1, "configuration itself must not rotate keys"
    result = provision(world)
    chain = parse(Chain, Path(result["custody_chain"]).read_bytes())
    assert len(chain.transactions) == 2
    head = chain.transactions[-1].completed.statement
    assert head.gpu.keys["node-a"].generation == 2
    assert head.gpu.keys["node-b"].generation == 1
    current = world.api.state["secrets"]["gpu"]["data"]
    assert current["node-a"] != old["node-a"]
    assert current["node-b"] == old["node-b"]
    assert current == world.api.state["secrets"]["cpu"]["data"]
    request = parse(ProvisionInputs, request_path.read_bytes())
    assert Path(request.retired_key_file).stat().st_mode & 0o777 == 0o600
    assert provision(world, probe=True) == result
    assert result["runtime_activation"] == "DEPLOYED_NOT_WITNESSED"
    assert world.activation_io.events == [
        "capture",
        "guard",
        "fence",
        "executor",
        "install",
        "cpu",
        "observe",
        "unfence",
    ]
    assert provision(world) == result
    assert world.helper_calls == 2
    assert result["runtime_activation"] == "DEPLOYED_NOT_WITNESSED"


@pytest.mark.parametrize(
    "failure",
    ["incomplete", "no-activation", "wrong-predecessor", "replay", "no-retired-file"],
)
def test_started_request_only_advances_from_its_complete_activated_predecessor(
    installed, failure
):
    world, chain_path, request_path = installed
    before = registration_path(world.state_dir).read_bytes()
    request = parse(ProvisionInputs, request_path.read_bytes())
    auth_path = Path(request.authorization)
    authorization = parse(Signed[Authorization], auth_path.read_bytes()).statement
    if failure == "incomplete":
        chain_path.unlink()
    elif failure == "no-activation":
        Path(request.previous_chain).write_bytes(chain_path.read_bytes())
    elif failure == "wrong-predecessor":
        authorization = authorization.model_copy(
            update={"previous_receipt_sha256": "f" * 64}
        )
    elif failure == "replay":
        authorization = authorization.model_copy(
            update={
                "transaction_id": load_admin_custody(world.state_dir).transactions[
                    world.gpu.eks_arn
                ]
            }
        )
    else:
        request = request.model_copy(update={"retired_key_file": None})
    auth_path.write_bytes(
        canonical(world.authorities.envelope(authorization, "approval"))
    )
    request_path.write_bytes(canonical(request))
    with pytest.raises(CustodyError):
        world.configure()
    assert registration_path(world.state_dir).read_bytes() == before
    assert world.helper_calls == 1


def test_incomplete_rotation_keeps_the_cas_intent_and_denies_shape_only_resume(
    installed,
):
    world, _, _ = installed
    world.configure()
    world.api.state["events"] = [
        {"on": "cpu:replace", "returncode": 1, "lost_ack": True}
    ]
    # A replace ACK can be recovered by the existing ownership/CAS protocol.
    result = provision(world)
    assert result["runtime_activation"] == "DEPLOYED_NOT_WITNESSED"
    path = Path(result["custody_chain"])
    path.unlink()
    with pytest.raises(
        CustodyReconciliationRequired, match="incomplete custody intent"
    ):
        provision(world)
    assert world.helper_calls == 2
