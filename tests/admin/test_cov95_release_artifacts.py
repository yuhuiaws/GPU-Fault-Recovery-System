from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import release_artifacts as artifacts
from gpu_fault.admin import release_postgres
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.postgres_grant import (
    ALLOCATION_ENV,
    LOCAL_DOCKER_HOST,
    POSTGRES_URL_ENV,
    postgres_test_environment,
)
from tests.admin._release_postgres_support import CID, FakePostgresDocker

REPOSITORY = "123456789012.dkr.ecr.us-east-1.amazonaws.com/example/runtime"
COMMIT = "a" * 40


class ReleaseTransport:
    def __init__(self, directory):
        self.root = directory / "repo"
        self.state = directory / "state"
        self.dist = self.root / "dist"
        self.dist.mkdir(parents=True)
        signing = self.state / "release-signing"
        signing.mkdir(parents=True)
        self.state.chmod(0o700)
        for name in ("cosign.key", "cosign.pub"):
            path = signing / name
            path.write_text("example-placeholder")
            path.chmod(0o600)
        self.calls = []
        self.legacy_calls = []
        self.gate = self.dist / "ci-gate.json"
        self.gate.write_text("{}")
        self.candidate = '{"available":false}'
        self.receipt = '{"available":false}'
        self.impact = '{"postgres":false}'
        self.git_error = False
        self.dirty = False
        self.ready = True
        self.postgres = FakePostgresDocker()
        self.during_build = None

    def manifest(self, *, schema=4, staging=False):
        names = [
            "runtime",
            "executor",
            "node_installer",
            "node_dependencies",
            "dcgm_exporter",
            "adot",
        ]
        return {
            "release_id": "release-example",
            "schema_version": schema,
            "deployable": True,
            "staging_only": staging,
            "delivery": {
                "images": {
                    name: {"reference": REPOSITORY + "@sha256:" + character * 64}
                    for name, character in zip(names, "abcdef", strict=True)
                }
            },
        }

    def write_release(self, *, staging=False):
        value = self.manifest(staging=staging)
        (self.dist / "current-release.json").write_text(json.dumps(value))
        (self.dist / "current-attestation.json").write_text(
            json.dumps(
                {
                    "source": {"dirty": False, "git_commit": COMMIT},
                    "impact_base": "origin/main",
                }
            )
        )
        (self.dist / "current-attestation.bundle.json").write_text("{}")
        return value

    def run(self, arguments, **options):
        self.calls.append((list(arguments), dict(options)))
        if arguments[:3] == ["docker", "--host", LOCAL_DOCKER_HOST]:
            self.postgres.ready = self.ready
            return self.postgres.run(arguments, **options)
        if arguments[0] == "make":
            if self.during_build is not None:
                self.during_build(options["env"])
            self.write_release(staging="release-build-staging" in arguments)
            return ""
        if arguments[:3] == ["aws", "ecr", "get-login-password"]:
            return "example-login-placeholder"
        if arguments[:2] == ["docker", "inspect"]:
            return "12345"
        if arguments[0] == "docker":
            return ""
        if "--print-config-digest-environment" in arguments:
            return json.dumps(
                {
                    "GPU_FAULT_PYTHON_STACK_TOOL": "py-spy",
                    "GPU_FAULT_QUIESCE_RESTORE_COMMAND": "/example/restore",
                }
            )
        if "-c" in arguments:
            return "a" * 64
        script = Path(arguments[1]).name
        if script == "select-affected-tests.py":
            return self.impact
        if script == "restore_ci_candidate.py":
            return self.candidate
        if script == "ci_candidate_receipt.py":
            return self.receipt
        assert script == "verify-release-attestation.py", "unexpected release helper"
        return ""

    def legacy(self, arguments, **_options):
        self.legacy_calls.append(list(arguments))
        if arguments[0] == "git":
            output = (
                "dirty"
                if arguments[1] == "status" and self.dirty
                else ""
                if arguments[1] == "status"
                else COMMIT
            )
            return subprocess.CompletedProcess(
                arguments,
                1 if self.git_error else 0,
                output,
                "example git error" if self.git_error else "",
            )
        assert arguments[0] == "docker", "unexpected legacy process"
        return subprocess.CompletedProcess(
            arguments, 0 if self.ready or arguments[1] == "rm" else 1, "", ""
        )

    def images(self, arguments, **_options):
        ids = [
            value.split("=", 1)[1]
            for value in arguments
            if value.startswith("imageDigest=")
        ]
        return subprocess.CompletedProcess(
            arguments,
            0,
            json.dumps(
                {
                    "imageDetails": [
                        {
                            "registryId": "123456789012",
                            "repositoryName": "example/runtime",
                            "imageDigest": value,
                        }
                        for value in ids
                    ]
                }
            ),
            "",
        )

    def build(self, **options):
        return artifacts.build_signed_release(
            self,
            repository_root=self.root,
            state_dir=self.state,
            region="us-east-1",
            runtime_repository=REPOSITORY,
            cache_repository=None,
            runtime_profile="hyperpod-v1",
            **options,
        )

    def reusable(self, **options):
        return artifacts.load_reusable_signed_release(
            self,
            repository_root=self.root,
            state_dir=self.state,
            region="us-east-1",
            runtime_repository=REPOSITORY,
            runtime_profile="hyperpod-v1",
            staging_only=False,
            impact_base="origin/main",
            **options,
        )


def _tools_venv(root: Path) -> Path:
    """A supply-chain tools venv with the two executables the gates resolve."""
    (root / "bin").mkdir(parents=True)
    for name in ("python", "promtool"):
        path = root / "bin" / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    return root


@pytest.fixture
def transport(tmp_path, monkeypatch):
    value = ReleaseTransport(tmp_path)
    # The site's own toolchain is what a deploy binds the gates to; the
    # Makefile default must not leak in from the developer host.
    _tools_venv(value.state / artifacts.SITE_TOOLCHAIN_DIRECTORY)
    monkeypatch.setattr(
        artifacts, "SUPPLY_CHAIN_TOOLS_DEFAULT", tmp_path / "no-default-tools"
    )
    monkeypatch.delenv(artifacts.SUPPLY_CHAIN_PYTHON_ENV, raising=False)
    monkeypatch.delenv(artifacts.PROMTOOL_ENV, raising=False)
    monkeypatch.setattr(
        artifacts,
        "subprocess",
        SimpleNamespace(**{**vars(subprocess), "run": value.legacy}),
    )
    monkeypatch.setattr(artifacts, "bounded_command", value.images)
    monkeypatch.setattr(
        release_postgres, "time", SimpleNamespace(sleep=lambda _seconds: None)
    )
    monkeypatch.setattr(
        release_postgres,
        "secrets",
        SimpleNamespace(token_urlsafe=lambda _size: "example-placeholder"),
    )
    return value


@pytest.mark.parametrize("staging", [False, True])
def test_release_build_runs_only_fake_tools_then_verifies_artifact(transport, staging):
    result = transport.build(staging_only=staging)
    assert result["release_reused"] is False
    assert result["release_source"] == (
        "staging_impact" if staging else "local_full_gate"
    )
    assert result["agent_config_digest"] == "a" * 64
    targets = [
        arguments for arguments, _options in transport.calls if arguments[0] == "make"
    ]
    assert len(targets) == 1
    assert ("release-build-staging" if staging else "release-build") in targets[0]
    assert any(
        Path(arguments[1]).name == "verify-release-attestation.py"
        for arguments, _options in transport.calls
        if arguments[0] != "make"
    ), "release build omitted final attestation verification"
    if not staging:
        assert transport.postgres.removed == [CID]
        assert transport.postgres.directory is not None
        assert not transport.postgres.directory.exists(), (
            "release success retained PostgreSQL grant credentials"
        )


def test_explicit_tools_venv_is_located_by_its_symlink_not_its_target(tmp_path):
    """A venv's ``bin/python`` is a symlink chain ending outside the venv.

    2026-09-18: the deploy bound SUPPLY_CHAIN_PYTHON to ``<state-dir>/toolchain/
    bin/python`` and the release gate's own tests then resolved that symlink to
    ``/usr/bin/python3.12``, looked for ``/usr/bin/promtool`` and failed the gate
    with "names a venv without an executable bin/python and bin/promtool". The
    venv is where the symlink lives, not where it points.
    """
    interpreter = tmp_path / "interpreters" / "python3.12"
    interpreter.parent.mkdir()
    interpreter.write_text("#!/bin/sh\nexit 0\n")
    interpreter.chmod(0o755)
    venv = tmp_path / "toolchain"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python3.12").symlink_to(interpreter)
    (venv / "bin" / "python").symlink_to("python3.12")
    promtool = venv / "bin" / "promtool"
    promtool.write_text("#!/bin/sh\nexit 0\n")
    promtool.chmod(0o755)
    bound = artifacts.supply_chain_tools_environment(
        {artifacts.SUPPLY_CHAIN_PYTHON_ENV: str(venv / "bin" / "python")},
        state_dir=tmp_path / "state",
    )
    assert bound[artifacts.PROMTOOL_ENV] == str(promtool), (
        "promtool is the sibling of the symlink, never of its resolved target"
    )
    assert bound[artifacts.SUPPLY_CHAIN_PYTHON_ENV] == str(venv / "bin" / "python"), (
        "the operator's explicit interpreter path passes through unchanged"
    )


def test_release_gates_bind_supply_chain_tools_to_the_site(tmp_path, monkeypatch):
    """PROMTOOL comes from the site, not from whatever the operator's shell exported.

    2026-09-17: a fresh shell's `deploy --state-dir` failed `make promtool-preflight`
    although `<state-dir>/toolchain` held the pinned promtool -- only the previous
    operator's shell had exported SUPPLY_CHAIN_PYTHON.
    """
    state = tmp_path / "state"
    site = _tools_venv(state / artifacts.SITE_TOOLCHAIN_DIRECTORY)
    monkeypatch.setattr(
        artifacts, "SUPPLY_CHAIN_TOOLS_DEFAULT", tmp_path / "absent-default"
    )
    bound = artifacts.supply_chain_tools_environment(
        {"PATH": "/usr/bin"}, state_dir=state
    )
    assert bound["PATH"] == "/usr/bin", "unrelated variables must pass through"
    assert bound[artifacts.SUPPLY_CHAIN_PYTHON_ENV] == str(site / "bin" / "python")
    assert bound[artifacts.PROMTOOL_ENV] == str(site / "bin" / "promtool")
    # An operator's explicit tools venv wins over the site's.
    other = _tools_venv(tmp_path / "other")
    explicit = artifacts.supply_chain_tools_environment(
        {artifacts.SUPPLY_CHAIN_PYTHON_ENV: str(other / "bin" / "python")},
        state_dir=state,
    )
    assert explicit[artifacts.SUPPLY_CHAIN_PYTHON_ENV] == str(other / "bin" / "python")
    assert explicit[artifacts.PROMTOOL_ENV] == str(other / "bin" / "promtool")
    # Without a site toolchain the Makefile default (make ci-supply-chain-tools) serves.
    default = _tools_venv(tmp_path / "default-tools")
    monkeypatch.setattr(artifacts, "SUPPLY_CHAIN_TOOLS_DEFAULT", default)
    fallback = artifacts.supply_chain_tools_environment({}, state_dir=tmp_path / "bare")
    assert fallback[artifacts.SUPPLY_CHAIN_PYTHON_ENV] == str(
        default / "bin" / "python"
    )
    # Nothing usable: fail before the gates, naming the setup command and the site path.
    monkeypatch.setattr(artifacts, "SUPPLY_CHAIN_TOOLS_DEFAULT", tmp_path / "absent")
    with pytest.raises(BootstrapError, match="ci-supply-chain-tools") as missing:
        artifacts.supply_chain_tools_environment({}, state_dir=tmp_path / "bare")
    assert str(tmp_path / "bare" / artifacts.SITE_TOOLCHAIN_DIRECTORY) in str(
        missing.value
    )
    # An explicit venv without promtool is refused, never silently replaced.
    broken = tmp_path / "broken"
    (broken / "bin").mkdir(parents=True)
    python = broken / "bin" / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    with pytest.raises(BootstrapError, match="promtool"):
        artifacts.supply_chain_tools_environment(
            {artifacts.SUPPLY_CHAIN_PYTHON_ENV: str(python)}, state_dir=state
        )


def test_release_build_hands_the_site_toolchain_to_selection_and_make(transport):
    transport.build(staging_only=True)
    site = transport.state / artifacts.SITE_TOOLCHAIN_DIRECTORY
    bound = [
        options["env"]
        for arguments, options in transport.calls
        if arguments[0] == "make"
        or (
            len(arguments) > 1 and Path(arguments[1]).name == "select-affected-tests.py"
        )
    ]
    assert len(bound) == 2, "impact selection and the release build both run gates"
    assert all(
        env[artifacts.SUPPLY_CHAIN_PYTHON_ENV] == str(site / "bin" / "python")
        and env[artifacts.PROMTOOL_ENV] == str(site / "bin" / "promtool")
        for env in bound
    ), "every gate-running command must receive the site-bound tools"


def test_reusable_split_release_proves_every_signed_image(transport):
    transport.write_release()
    assert transport.reusable()["release_reused"] is True
    assert not any(
        arguments[0] in {"make", "docker", "aws"}
        for arguments, _options in transport.calls
    ), "reusable release started a build or registry login"


@pytest.mark.parametrize(
    "kind",
    ["missing-attestation", "bad-json", "not-object", "source-dirty", "tier", "commit"],
)
def test_reuse_rejects_unbound_inputs_without_building(transport, kind):
    value = transport.write_release()
    manifest = transport.dist / "current-release.json"
    attestation = transport.dist / "current-attestation.json"
    if kind == "missing-attestation":
        attestation.unlink()
    elif kind == "bad-json":
        manifest.write_text("invalid")
    elif kind == "not-object":
        manifest.write_text("[]")
    elif kind == "source-dirty":
        attestation.write_text('{"source":{"dirty":true}}')
    elif kind == "tier":
        value["staging_only"] = "false"
        manifest.write_text(json.dumps(value))
    else:
        attestation.write_text('{"source":{"dirty":false,"git_commit":"other"}}')
    assert transport.reusable() is None
    assert not transport.calls, "unbound reusable release reached helper commands"


@pytest.mark.parametrize("output", ["invalid", "[]", "{}", '{"postgres":"false"}'])
def test_staging_impact_plan_requires_boolean_postgres_selection(transport, output):
    transport.impact = output
    with pytest.raises(BootstrapError, match="invalid|incomplete"):
        artifacts.prepare_staging_impact_plan(
            transport, repository_root=transport.root, impact_base="origin/main"
        )
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "operation",
    [artifacts.restore_main_ci_candidate, artifacts.load_verified_ci_candidate_receipt],
)
@pytest.mark.parametrize(
    "scenario",
    ["unavailable", "invalid-json", "invalid-shape", "external", "missing", "valid"],
)
def test_ci_candidate_receipt_and_restore_bind_gate_to_dist(
    transport, operation, scenario
):
    if operation is artifacts.load_verified_ci_candidate_receipt:
        receipt = transport.state / "ci-candidates" / COMMIT / "verification.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text("{}")
    output = {
        "unavailable": '{"available":false}',
        "invalid-json": "invalid",
        "invalid-shape": '{"available":1}',
        "external": json.dumps(
            {"available": True, "ci_gate": str(transport.state / "outside.json")}
        ),
        "missing": json.dumps(
            {"available": True, "ci_gate": str(transport.dist / "missing.json")}
        ),
        "valid": json.dumps({"available": True, "ci_gate": str(transport.gate)}),
    }[scenario]
    transport.candidate = transport.receipt = output
    options = {"repository_root": transport.root}
    if operation is artifacts.load_verified_ci_candidate_receipt:
        options["state_dir"] = transport.state
    if scenario not in {"unavailable", "valid"}:
        with pytest.raises(
            BootstrapError,
            match="invalid JSON|incomplete|leaves repository dist|missing",
        ):
            operation(transport, **options)
    else:
        result = operation(transport, **options)
        assert result == (None if scenario == "unavailable" else json.loads(output))


@pytest.mark.parametrize(
    "kind", ["missing-key", "public-key-file", "public-password-file"]
)
def test_build_rejects_missing_or_public_signing_inputs_before_login(transport, kind):
    key = transport.state / "release-signing/cosign.key"
    if kind == "missing-key":
        key.unlink()
    elif kind == "public-key-file":
        key.chmod(0o644)
    else:
        password = key.with_name("cosign.password")
        password.write_text("example-placeholder")
        password.chmod(0o644)
    with pytest.raises(BootstrapError, match="missing|accessible"):
        transport.build(staging_only=True)
    assert not any(
        arguments[0] in {"aws", "docker", "make"}
        for arguments, _options in transport.calls
    ), "invalid signing inputs reached login or build"


@pytest.mark.parametrize("kind", ["attestation", "bundle", "public-key"])
def test_signature_validation_requires_all_input_files(transport, kind):
    transport.write_release()
    path = {
        "attestation": transport.dist / "current-attestation.json",
        "bundle": transport.dist / "current-attestation.bundle.json",
        "public-key": transport.state / "release-signing/cosign.pub",
    }[kind]
    path.unlink()
    with pytest.raises(BootstrapError, match="missing"):
        artifacts.verify_prebuilt_release(
            transport, repository_root=transport.root, state_dir=transport.state
        )
    assert transport.calls == []


@pytest.mark.parametrize(
    "scenario", ["missing", "bad-json", "schema", "tier", "images", "staging"]
)
def test_prebuilt_release_rejects_incomplete_or_unapproved_identity(
    transport, scenario
):
    value = transport.write_release()
    path = transport.dist / "current-release.json"
    if scenario == "missing":
        path.unlink()
    elif scenario == "bad-json":
        path.write_text("invalid")
    else:
        if scenario == "schema":
            value["schema_version"] = 5
        elif scenario == "tier":
            value["staging_only"] = "false"
        elif scenario == "images":
            value["delivery"]["images"]["runtime"] = {"reference": "unversioned"}
        else:
            value["staging_only"] = True
        path.write_text(json.dumps(value))
    with pytest.raises(
        BootstrapError,
        match="missing|invalid|schema version|boolean|incomplete|authorization",
    ):
        artifacts.load_prebuilt_release(
            transport, repository_root=transport.root, runtime_profile="hyperpod-v1"
        )
    assert transport.calls == []


def test_failed_temporary_postgres_probe_cleans_up_its_fake_container(transport):
    transport.ready = False
    with pytest.raises(BootstrapError, match="did not become ready"):
        with artifacts.isolated_postgres_url(transport):
            pytest.fail("unready fake database was accepted")
    assert transport.postgres.readiness_calls == 60
    assert transport.postgres.removed == [CID]


def test_body_failure_also_cleans_up_owned_fake_postgres(transport):
    with pytest.raises(RuntimeError, match="consumer failed"):
        with artifacts.isolated_postgres_url(transport) as url:
            assert url == "postgresql://postgres@127.0.0.1:54321/postgres", (
                "temporary database reference escaped the fake loopback binding"
            )
            raise RuntimeError("consumer failed")
    assert transport.postgres.removed == [CID]


def test_unconfirmed_postgres_cleanup_fails_the_build_before_final_verification(
    transport,
):
    transport.postgres.keep_after_remove = True
    with pytest.raises(release_postgres.PostgresCleanupError, match="unconfirmed"):
        transport.build()
    assert transport.postgres.containers, "test did not retain its simulated database"
    assert not any(
        len(arguments) > 1
        and Path(arguments[1]).name == "verify-release-attestation.py"
        for arguments, _options in transport.calls
    ), "the build accepted artifacts after unconfirmed PostgreSQL cleanup"
    original = list(transport.calls)
    with pytest.raises(
        release_postgres.PostgresCleanupError, match="explicit cleanup reconciliation"
    ):
        transport.build()
    assert transport.calls == original, (
        "artifact reuse bypassed the unresolved local database lifecycle"
    )


def test_real_build_handoff_passes_grant_reference_without_changing_build_home(
    transport, monkeypatch
):
    monkeypatch.setenv("HOME", "/original/build-home")
    monkeypatch.setenv("PGPASSFILE", "/original/pgpass")
    observed = []

    def during_build(environment):
        observed.append(dict(environment))
        assert environment["HOME"] == "/original/build-home"
        assert environment["PGPASSFILE"] == "/original/pgpass"
        assert environment[POSTGRES_URL_ENV] == (
            "postgresql://postgres@127.0.0.1:54321/postgres"
        )
        directory = Path(environment[ALLOCATION_ENV])
        assert directory == transport.state / release_postgres.ALLOCATION_DIRECTORY
        child = postgres_test_environment(environment)
        assert child["HOME"] == str(directory / "pgpass-home")
        assert child["PGPASSFILE"] == str(directory / "pgpass-home/.pgpass")
        assert "COSIGN_PASSWORD" not in child, (
            "PostgreSQL test environment received the release signing password"
        )

    transport.during_build = during_build
    assert transport.build()["release_source"] == "local_full_gate"
    assert len(observed) == 1
    assert transport.postgres.removed == [CID]


def test_git_probe_failure_is_not_a_clean_build(transport):
    transport.git_error = True
    with pytest.raises(BootstrapError, match="git command failed"):
        transport.build()
    assert transport.calls == []


def test_empty_staging_base_is_rejected_before_every_process(transport):
    with pytest.raises(BootstrapError, match="non-empty impact base"):
        transport.build(staging_only=True, impact_base=" ")
    assert transport.calls == transport.legacy_calls == []


@pytest.mark.parametrize(
    "error",
    [
        OSError("example error"),
        UnicodeError("example text"),
        subprocess.TimeoutExpired(["fake"], 1),
    ],
)
def test_runtime_image_transport_failure_is_explicit(transport, monkeypatch, error):
    def failed(_arguments):
        raise error

    monkeypatch.setattr(artifacts, "bounded_command", failed)
    with pytest.raises(BootstrapError, match="cannot verify signed runtime image"):
        artifacts.runtime_image_exists(
            region="us-east-1", reference=REPOSITORY + "@sha256:" + "a" * 64
        )
