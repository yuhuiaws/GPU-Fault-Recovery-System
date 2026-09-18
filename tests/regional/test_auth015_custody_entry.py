from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import acceptance_supervision
from scripts.e2e.regional import identity_acceptance_auth as auth
from scripts.e2e.regional import live_driver_guard as guard
from scripts.e2e.regional import run_identity_acceptance as entry
from tests.regional._auth015_custody_support import RuntimeFixture
from tests.regional._cov95_identity_support import offline_guard as offline_guard

CASE = "GF-REGIONAL-AUTH-015"


@pytest.fixture
def caller(tmp_path, monkeypatch):
    runtime = RuntimeFixture(tmp_path, monkeypatch)
    runtime.install_runtime_keys()
    runtime.witness()
    runtime.rotate()
    runtime.install_runtime_keys()
    inputs = runtime.inputs()
    # The entry hashes the site inputs into the plan environment and pins
    # both kubeconfigs there, exactly as ``IdentitySite`` exposes them.
    site_file = tmp_path / "site"
    site_file.write_text("synthetic: site\n", encoding="utf-8")
    cpu_kubeconfig = tmp_path / "cpu.kubeconfig"
    gpu_kubeconfig = tmp_path / "gpu.kubeconfig"
    cpu_kubeconfig.write_text("unit CPU connection fixture\n", encoding="utf-8")
    gpu_kubeconfig.write_text("unit GPU connection fixture\n", encoding="utf-8")
    proxy = SimpleNamespace(
        cpu_kubeconfig=cpu_kubeconfig,
        gpu_kubeconfig=gpu_kubeconfig,
        target=lambda _: runtime.site.target,
        regional=lambda _: SimpleNamespace(
            evidence_identity=lambda: {
                "release_id": "release-a",
                "cluster_id": "cluster-a",
            }
        ),
    )
    monkeypatch.setattr(entry, "IdentitySite", lambda _: proxy)
    monkeypatch.setattr(entry, "install_site_profile", lambda: None)
    monkeypatch.setattr(entry, "predecessor_path", lambda *a: (None, None))
    monkeypatch.setattr(entry, "auth015_focused_tests", lambda: {"passed": True})
    monkeypatch.setattr(
        entry,
        "record_focused_tests",
        lambda details, value: details.update(focused_tests=value),
    )
    monkeypatch.setattr(entry, "reusable_focused_tests", lambda _: {"passed": True})
    monkeypatch.setattr(guard, "source_digest", lambda: "a" * 64)
    monkeypatch.setattr(guard, "applied_site_profile", lambda: None)
    monkeypatch.setattr(
        guard,
        "current_acceptance_scope",
        lambda: SimpleNamespace(
            plan_fields=lambda: {"execution_scope": "offline-custody-test"}
        ),
    )
    monkeypatch.setattr(
        acceptance_supervision, "bind_command_supervision", lambda _: None
    )
    calls = []

    def handler(_site, target, **kwargs):
        calls.append(kwargs)
        return auth.run_auth015(runtime.site, target, **kwargs)

    monkeypatch.setattr(entry, "run_auth015", handler)
    run_dir = tmp_path / "run"
    argv = [
        "identity",
        "--run-dir",
        str(run_dir),
        "--site",
        str(site_file),
        "--case",
        CASE,
        "--cluster-id",
        "cluster-a",
        "--node",
        "node-a",
        "--node",
        "node-b",
        "--auth015-release-proof",
        str(runtime.site.files.descriptor_path),
        "--auth015-custody-proof",
        str(inputs.descriptor_path),
        "--auth015-custody-trust-sha256",
        inputs.trust_sha256,
        "--maintenance-window-end",
        (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        yield SimpleNamespace(
            runtime=runtime,
            inputs=inputs,
            argv=argv,
            run_dir=run_dir,
            calls=calls,
            proxy=proxy,
        )
    finally:
        runtime.close()


def execute(caller):
    caller.argv.extend(["--execute", "--confirm", "AUTH015_EXECUTE"])
    return entry.main()


def test_guarded_entry_binds_custody_inputs_and_dispatches_real_readonly_witness(
    caller,
):
    writes = list(caller.runtime.provisioning.events)
    assert entry.main() == 0
    plan = json.loads((caller.run_dir / "cases" / CASE / "plan.json").read_text())
    binding = plan["details"]["custody_inputs"]
    assert plan["environment"]["AUTH015_CUSTODY_INPUTS_SHA256"] == guard.details_sha256(
        binding
    )
    assert caller.calls == []
    assert execute(caller) == 0
    assert caller.calls[0]["custody_inputs"].expected_identity == binding
    result = json.loads((caller.run_dir / "cases" / CASE / f"{CASE}.json").read_text())
    assert result["verdict"] == "PASS"
    assert result["checks"]["independent_signed_activation_receipt"] is True
    assert result["activation_receipt"]["statement"]["retired_key_denied"] is True
    assert caller.runtime.provisioning.events == writes


@pytest.mark.parametrize("which", ["descriptor", "chain", "trust", "public", "retired"])
def test_same_path_custody_input_replacement_invalidates_the_approved_plan(
    caller, which
):
    assert entry.main() == 0
    fixture = caller.runtime.provisioning
    path = {
        "descriptor": caller.inputs.descriptor_path,
        "chain": caller.runtime.root / "custody-chain.json",
        "trust": fixture.authorities.trust_path,
        "public": caller.runtime.root / "witness.pem",
        "retired": caller.runtime.retired_path,
    }[which]
    path.write_bytes(path.read_bytes() + b"\n")
    if which == "trust":
        caller.argv[caller.argv.index("--auth015-custody-trust-sha256") + 1] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )
    with pytest.raises(RuntimeError, match="drifted at environment"):
        execute(caller)
    assert caller.calls == []


def test_post_authorization_input_drift_is_refused_inside_the_witness(
    caller, monkeypatch
):
    assert entry.main() == 0
    original = entry.authorize_execution

    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        path = caller.runtime.root / "custody-chain.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(entry, "authorize_execution", change)
    assert execute(caller) == 1
    result = json.loads((caller.run_dir / "cases" / CASE / f"{CASE}.json").read_text())
    assert result["verdict"] == "FAIL"
    assert result["partial"]["requires_new_authorized_evidence"] is True
    assert "approved plan" in result["error"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"fleet_master_file": Path("/not-opened")},
        {"host_probe_image": "not-used"},
        {"auth015_release_proof": None},
        {"auth015_custody_proof": None},
        {"auth015_custody_trust_sha256": ""},
        {"node": ["node-a", "node-a"]},
        {"node": ["node-a"]},
    ],
)
def test_custody_entry_requires_complete_explicit_inputs_without_master_access(
    caller, mutation
):
    arguments = entry.parser().parse_args(caller.argv[1:])
    for name, value in mutation.items():
        setattr(arguments, name, value)
    with pytest.raises(entry.IdentityAcceptanceError, match="custody requires"):
        entry.validate_case_arguments(arguments, caller.proxy)
    assert caller.calls == []


def test_custody_flags_cannot_be_reused_for_another_case(caller):
    arguments = entry.parser().parse_args(caller.argv[1:])
    arguments.case = "GF-REGIONAL-AUTH-010"
    with pytest.raises(entry.IdentityAcceptanceError, match="only valid"):
        entry.validate_case_arguments(arguments, caller.proxy)


def test_foreign_custody_target_never_creates_an_approvable_plan(caller, monkeypatch):
    original = entry.verify_custody_inputs

    def foreign(inputs):
        chain, head, crypto = original(inputs)
        binding = head.authorization.statement.binding
        authorization = head.authorization.statement.model_copy(
            update={
                "binding": binding.model_copy(
                    update={
                        "site": binding.site.model_copy(update={"cluster_id": "other"})
                    }
                )
            }
        )
        return (
            chain,
            head.model_copy(
                update={
                    "authorization": head.authorization.model_copy(
                        update={"statement": authorization}
                    )
                }
            ),
            crypto,
        )

    monkeypatch.setattr(entry, "verify_custody_inputs", foreign)
    with pytest.raises(entry.IdentityAcceptanceError, match="different target"):
        entry.main()
    assert caller.calls == []
    assert not (caller.run_dir / "cases" / CASE / "plan.json").exists(), (
        "foreign custody target must not produce an approvable plan"
    )


@pytest.mark.parametrize("returncode", [0, 1])
def test_default_focused_precheck_includes_crypto_provisioning_and_activation(
    monkeypatch, returncode
):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs == {"check": False, "timeout": 600}
        return subprocess.CompletedProcess(command, returncode, "", "")

    monkeypatch.setattr(auth, "run", run)
    result = auth.auth015_focused_tests()
    assert result["passed"] is (returncode == 0)
    assert result["returncode"] == returncode
    assert commands == [result["command"]]
    assert {item.split("::")[0] for item in result["command"][4:]} == {
        "tests/fleet/test_fleet.py",
        "tests/admin/test_node_key_custody_openssl.py",
        "tests/deploy/test_node_key_custody_provisioning.py",
        "tests/regional/test_auth015_custody_activation.py",
    }
