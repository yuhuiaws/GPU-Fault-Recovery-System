from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import acceptance_supervision
from scripts.e2e.regional import live_driver_guard as guard
from scripts.e2e.regional import run_identity_acceptance as entry
from scripts.e2e.regional.auth015_protocol import Auth015ProofError
from scripts.e2e.regional.auth015_release import verify_release_inputs
from tests.regional._cov95_auth015_release import ReleaseFiles
from tests.regional._cov95_identity_support import offline_guard as offline_guard

CASE_ID = "GF-REGIONAL-AUTH-015"


@pytest.fixture
def caller(tmp_path, monkeypatch):
    files = ReleaseFiles(tmp_path, monkeypatch)
    master = tmp_path / "synthetic-master"
    master.write_text("synthetic-master-" + "m" * 48)
    master.chmod(0o600)
    # The entry hashes the site inputs into the plan environment and pins
    # both kubeconfigs there, exactly as ``IdentitySite`` exposes them.
    site_file = tmp_path / "site.yaml"
    site_file.write_text("synthetic: site\n", encoding="utf-8")
    cpu_kubeconfig = tmp_path / "cpu.kubeconfig"
    gpu_kubeconfig = tmp_path / "gpu.kubeconfig"
    cpu_kubeconfig.write_text("unit CPU connection fixture\n", encoding="utf-8")
    gpu_kubeconfig.write_text("unit GPU connection fixture\n", encoding="utf-8")
    target = SimpleNamespace(cluster_id="cluster-a", context="context-a")
    site = SimpleNamespace(
        cpu_kubeconfig=cpu_kubeconfig,
        gpu_kubeconfig=gpu_kubeconfig,
        target=lambda _: target,
        regional=lambda _: SimpleNamespace(
            evidence_identity=lambda: {
                "cluster_id": "cluster-a",
                "release_id": "release-a",
            }
        ),
    )
    monkeypatch.setattr(entry, "IdentitySite", lambda _: site)
    monkeypatch.setattr(entry, "install_site_profile", lambda: None)
    monkeypatch.setattr(entry, "predecessor_path", lambda *args: (None, None))
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
            plan_fields=lambda: {"execution_scope": "offline-test"}
        ),
    )
    monkeypatch.setattr(
        acceptance_supervision, "bind_command_supervision", lambda _: None
    )
    calls = []

    def handler(*args, **kwargs):
        calls.append(kwargs)
        verify_release_inputs(kwargs["release_inputs"])
        return {
            "verdict": "FAIL",
            "not_evaluated": {
                "installation_time_master_custody": "not proved",
                "deployed_node_a_key_activation": "not proved",
            },
        }

    monkeypatch.setattr(entry, "run_auth015", handler)
    argv = [
        "identity",
        "--run-dir",
        str(tmp_path),
        "--site",
        str(site_file),
        "--case",
        CASE_ID,
        "--cluster-id",
        "cluster-a",
        "--node",
        "node-a",
        "--node",
        "node-b",
        "--fleet-master-file",
        str(master),
        "--host-probe-image",
        "test@sha256:" + "a" * 64,
        "--auth015-release-proof",
        str(files.descriptor_path),
        "--maintenance-window-end",
        (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return SimpleNamespace(files=files, argv=argv, calls=calls, root=tmp_path)


def execute(caller):
    caller.argv.extend(["--execute", "--confirm", "AUTH015_EXECUTE"])
    return entry.main()


def test_plan_records_content_binding_and_execution_receives_that_frozen_identity(
    caller,
):
    assert entry.main() == 0
    plan = json.loads((caller.root / "cases" / CASE_ID / "plan.json").read_text())
    binding = plan["details"]["deployed_protocol_release_inputs"]
    assert plan["environment"]["AUTH015_RELEASE_INPUTS_SHA256"] == guard.details_sha256(
        binding
    )
    assert set(binding) == {
        "descriptor",
        "attestation",
        "signature_bundle",
        "public_key",
        "manifest",
        "release_id",
    }
    assert caller.calls == [], "planning must never send a Node Agent challenge"
    assert execute(caller) == 1, (
        "a protocol input does not remove the remaining custody gaps"
    )
    assert caller.calls[0]["release_inputs"].expected_identity == binding


@pytest.mark.parametrize(
    "file", ["descriptor", "attestation", "bundle", "manifest", "public-key"]
)
def test_same_path_changed_release_bytes_cannot_reuse_a_plan(caller, file):
    assert entry.main() == 0
    files = caller.files
    if file == "public-key":
        files.public_key_path.write_bytes(files.public_key_path.read_bytes() + b"\n")
        files.descriptor["cosign_public_key_sha256"] = hashlib.sha256(
            files.public_key_path.read_bytes()
        ).hexdigest()
        files.write_descriptor()
    elif file == "manifest":
        files.manifest_path.write_bytes(files.manifest_path.read_bytes() + b"\n")
        attestation = json.loads(files.attestation_path.read_text())
        attestation["subject"]["manifest_sha256"] = hashlib.sha256(
            files.manifest_path.read_bytes()
        ).hexdigest()
        files.attestation_path.write_text(json.dumps(attestation))
    else:
        path = {
            "descriptor": files.descriptor_path,
            "attestation": files.attestation_path,
            "bundle": files.bundle_path,
        }[file]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="drifted at environment"):
        execute(caller)
    assert caller.calls == [], (
        "changed release bytes must stop before dispatch to the AUTH015 handler"
    )


def test_release_change_after_authorization_is_refused_inside_the_proofer(
    caller, monkeypatch
):
    assert entry.main() == 0
    authorize = entry.authorize_execution

    def change(*args, **kwargs):
        deadline = authorize(*args, **kwargs)
        caller.files.bundle_path.write_bytes(b"{} \n")
        return deadline

    monkeypatch.setattr(entry, "authorize_execution", change)
    assert execute(caller) == 1
    result = json.loads(
        (caller.root / "cases" / CASE_ID / f"{CASE_ID}.json").read_text()
    )
    assert "differ from the approved plan" in result["error"]
    assert result["verdict"] == "FAIL"


def test_other_cases_cannot_accept_the_auth015_proof_flag(caller):
    arguments = entry.parser().parse_args(caller.argv[1:])
    arguments.case = "GF-REGIONAL-AUTH-010"
    with pytest.raises(entry.IdentityAcceptanceError, match="only valid for AUTH-015"):
        entry.validate_case_arguments(
            arguments,
            SimpleNamespace(target=lambda _: SimpleNamespace(cluster_id="cluster-a")),
        )
    assert caller.files.commands == [], (
        "a foreign case must not even invoke release verification"
    )


def test_bad_signature_or_missing_proof_file_cannot_produce_a_passing_plan(caller):
    caller.files.bundle_path.unlink()
    with pytest.raises(Auth015ProofError):
        entry.main()
    assert caller.calls == []
    assert not (caller.root / "cases" / CASE_ID / "plan.json").exists(), (
        "an incomplete signed-release input must not produce an approvable plan"
    )


@pytest.mark.parametrize("case_id", [CASE_ID, "GF-REGIONAL-AUTH-014"])
@pytest.mark.parametrize("failure", ["abort", "predecessor", "unauthorized"])
def test_authorized_attempt_fences_old_pass_before_any_fallible_dispatch(
    caller, monkeypatch, case_id, failure
):
    if case_id != CASE_ID:
        caller.argv[caller.argv.index("--case") + 1] = case_id
        flag = caller.argv.index("--auth015-release-proof")
        del caller.argv[flag : flag + 2]
    caller.argv.extend(["--attempt", "2"])
    assert entry.main() == 0
    path = caller.root / "cases" / case_id / f"{case_id}.json"
    old = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": case_id,
        "attempt": 1,
        "verdict": "PASS",
        "cluster_id": "cluster-a",
        "release_id": "release-a",
    }
    path.write_text(json.dumps(old))
    confirmation = case_id.removeprefix("GF-REGIONAL-").replace("-", "") + "_EXECUTE"
    caller.argv.extend(["--execute", "--confirm", confirmation])

    class AuthorizedAbort(BaseException):
        pass

    if failure == "unauthorized":
        caller.argv[caller.argv.index("--maintenance-window-end") + 1] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        with pytest.raises(RuntimeError, match="window has ended"):
            entry.main()
        assert json.loads(path.read_text()) == old, (
            "a request that never obtained authorization must not overwrite prior evidence"
        )
        return
    if failure == "predecessor":
        monkeypatch.setattr(
            entry, "predecessor_path", lambda *args: ("prior", caller.root / "prior")
        )
        monkeypatch.setattr(
            entry, "predecessor_evidence", lambda *a, **k: {"valid": False}
        )
        with pytest.raises(entry.IdentityAcceptanceError, match="predecessor"):
            entry.main()
    else:

        def abort(*args, **kwargs):
            raise AuthorizedAbort("synthetic authorized abort")

        monkeypatch.setattr(
            entry, "run_auth015" if case_id == CASE_ID else "run_auth014", abort
        )
        with pytest.raises(AuthorizedAbort):
            entry.main()
    current = json.loads(path.read_text())
    assert current["verdict"] != "PASS", (
        "an interrupted current attempt must not retain an earlier PASS"
    )
    assert current["attempt"] == 2
    assert current["case_id"] == case_id
    assert current["cluster_id"] == "cluster-a" and current["release_id"] == "release-a"
