from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault.admin.node_key_custody import ProvisionInputs
from gpu_fault.admin.node_key_custody_admin import CustodyPreparationRequired
from gpu_fault.admin.node_key_custody_admin_config import (
    configure_admin_custody,
    load_admin_custody,
    registration_path,
)
from gpu_fault.admin.node_key_custody_crypto import parse
from gpu_fault.admin.node_key_custody_models import (
    Authorization,
    CustodyError,
    Signed,
    canonical,
)
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
    world.authorize()
    return world


def test_relative_selection_paths_are_resolved_and_registration_is_private(prepared):
    world = prepared
    config = json.loads(world.selection_file.read_text())
    config["trust"] = str(world.authorities.trust_path.relative_to(world.root))
    config["clusters"][world.gpu.eks_arn] = "request.json"
    world.selection_file.write_text(json.dumps(config))
    configure_admin_custody(
        world.state_dir, world.selection_file, world.authorities.trust_pin
    )
    registration = load_admin_custody(world.state_dir)
    assert registration.selection.trust == str(world.authorities.trust_path)
    assert registration.selection.clusters[world.gpu.eks_arn] == str(
        world.root / "request.json"
    )
    assert registration_path(world.state_dir).stat().st_mode & 0o777 == 0o600
    assert registration_path(world.state_dir).parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    "failure",
    ["remove-cluster", "remove-request", "state-dir", "snapshot", "pin", "directory"],
)
def test_configure_cannot_silently_downgrade_or_repair_damaged_enrollment(
    prepared, failure
):
    world = prepared
    before = registration_path(world.state_dir).read_bytes()
    if failure == "remove-cluster":
        world.requests = {world.gpu.eks_arn + "-other": None}
    elif failure == "remove-request":
        world.requests[world.gpu.eks_arn] = None
    elif failure in {"state-dir", "snapshot", "pin"}:
        registration = json.loads(before)
        if failure == "state-dir":
            registration["state_directory"] = str(world.root / "another-site")
        elif failure == "snapshot":
            registration["transactions"] = {}
        else:
            registration["trust_sha256"] = "0" * 64
        registration_path(world.state_dir).write_text(json.dumps(registration))
    else:
        registration_path(world.state_dir).parent.chmod(0o755)
    with pytest.raises(CustodyError):
        world.configure()
    assert world.helper_calls == 0
    assert world.api.state["write_attempts"] == []


def test_missing_registration_cannot_be_replaced_with_fresh_intent(prepared):
    world = prepared
    registration_path(world.state_dir).unlink()
    with pytest.raises(CustodyError, match="registration is missing"):
        world.configure()


@pytest.mark.parametrize("failure", ["cluster", "trust-source", "verifier"])
def test_enrollment_checks_independent_approval_and_exact_trust(prepared, failure):
    world = prepared
    before = registration_path(world.state_dir).read_bytes()
    arguments = {}
    if failure == "cluster":
        path = world.root / "approved.json"
        approved = parse(Signed[Authorization], path.read_bytes()).statement
        approved = approved.model_copy(
            update={
                "binding": approved.binding.model_copy(
                    update={
                        "site": approved.binding.site.model_copy(
                            update={"gpu_eks_arn": world.gpu.eks_arn + "-other"}
                        )
                    }
                )
            }
        )
        path.write_bytes(canonical(world.authorities.envelope(approved, "approval")))
    elif failure == "trust-source":
        other = world.root / "other-trust.json"
        other.write_bytes(world.authorities.trust_path.read_bytes())
        request_path = world.root / "request.json"
        request = parse(ProvisionInputs, request_path.read_bytes())
        request_path.write_bytes(
            canonical(request.model_copy(update={"trust": str(other)}))
        )
        for name in ("approval.pem", "provisioner.pem", "witness.pem"):
            (world.root / name).write_bytes(
                (world.authorities.root / name).read_bytes()
            )
    else:
        crypto = world.authorities.crypto()
        crypto.trust_sha256 = "0" * 64
        arguments["crypto"] = crypto
    with pytest.raises(CustodyError):
        configure_admin_custody(
            world.state_dir,
            world.selection_file,
            world.authorities.trust_pin,
            **arguments,
        )
    assert registration_path(world.state_dir).read_bytes() == before


def test_changed_unstarted_request_requires_and_accepts_explicit_reconfiguration(
    prepared,
):
    world = prepared
    original = load_admin_custody(world.state_dir)
    path = world.root / "request.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(CustodyError, match="inputs changed"):
        load_admin_custody(world.state_dir)
    world.configure()
    changed = load_admin_custody(world.state_dir)
    assert changed.input_sha256 != original.input_sha256
    assert changed.transactions == original.transactions
    assert provision(world)["runtime_activation"] == "NOT_PROVED"


def test_configure_rechecks_inputs_after_signature_verification(prepared, monkeypatch):
    world = prepared
    before = registration_path(world.state_dir).read_bytes()
    verifier = world.authorities.crypto()
    verify = verifier.verify

    def changed(envelope, role):
        result = verify(envelope, role)
        path = Path(world.requests[world.gpu.eks_arn])
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(verifier, "verify", changed)
    with pytest.raises(CustodyError, match="changed while configuring"):
        configure_admin_custody(
            world.state_dir,
            world.selection_file,
            world.authorities.trust_pin,
            crypto=verifier,
        )
    assert registration_path(world.state_dir).read_bytes() == before
