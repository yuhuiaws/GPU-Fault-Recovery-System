from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin.bootstrap_dependencies import (
    deploy_host_dependency_report,
    load_deploy_host_tool_manifest,
)
from scripts import deploy_host_bundle, setup_deploy_host

ROOT = Path(__file__).resolve().parents[1]


def test_deploy_host_lock_and_tool_manifest_are_complete() -> None:
    lock = (ROOT / "requirements/deploy-host.lock").read_text(encoding="utf-8")
    manifest = load_deploy_host_tool_manifest()
    names = {item["name"] for item in manifest["tools"]}

    assert "--hash=sha256:" in lock
    for package in ("build==", "boto3==", "pytest==", "ruff==", "yamllint=="):
        assert package in lock, f"deploy-host lock is missing {package}"
    assert {
        "aws",
        "cosign",
        "docker-daemon",
        "docker-buildx",
        "kubectl",
        "helm",
    } <= names


def _bundle_tree(tmp_path: Path) -> Path:
    root = tmp_path / deploy_host_bundle.BUNDLE_ROOT
    (root / "requirements").mkdir(parents=True)
    (root / "wheelhouse").mkdir()
    (root / "config").mkdir()
    (root / "requirements/deploy-host.lock").write_text(
        "example==1\n", encoding="utf-8"
    )
    wheel = root / "wheelhouse/example-1-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    wheel.chmod(0o644)
    template = root / "config/admin-config.example.yaml"
    template.write_text(
        "apiVersion: gpu-fault.aws/v1alpha1\nkind: AdminConfig\nspec: {}\n",
        encoding="utf-8",
    )
    compatibility = deploy_host_bundle.host_compatibility()
    deploy_host_bundle.write_bundle_manifest(
        root,
        metadata={
            "compatibility": compatibility,
            "payload_identity_sha256": "b" * 64,
            "platform_id": deploy_host_bundle.bundle_platform_id(compatibility),
            "project_version": "1.0",
            "project_wheel": wheel.relative_to(root).as_posix(),
            "python": "3.12",
            "requirements": {"deploy_host": "requirements/deploy-host.lock"},
            "source": {"git_commit": "a" * 40, "dirty": False},
            "tool_manifest": "deploy-host-tools.json",
            "admin_config_template": "config/admin-config.example.yaml",
        },
    )
    return root


def test_deploy_host_bundle_is_deterministic_and_verified(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    deploy_host_bundle.write_deterministic_archive(root, first)
    deploy_host_bundle.write_deterministic_archive(root, second)

    assert deploy_host_bundle.sha256_file(first) == deploy_host_bundle.sha256_file(
        second
    ), "identical deploy-host inputs produced different archives"
    extracted = deploy_host_bundle.extract_verified_bundle(
        first, tmp_path / "extracted"
    )
    assert deploy_host_bundle.verify_bundle_tree(extracted)["project_version"] == "1.0"
    assert (extracted / "config/admin-config.example.yaml").is_file(), (
        "deploy-host archive omitted the administrator config template"
    )


def test_deploy_host_dependency_identity_covers_locks_and_platform(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    requirements = root / "requirements"
    requirements.mkdir(parents=True)
    (requirements / "build.lock").write_text("build-a\n", encoding="utf-8")
    (requirements / "deploy-host.lock").write_text("host-a\n", encoding="utf-8")
    compatibility = deploy_host_bundle.host_compatibility()

    first = deploy_host_bundle.dependency_identity(root, compatibility)
    (requirements / "deploy-host.lock").write_text("host-b\n", encoding="utf-8")
    second = deploy_host_bundle.dependency_identity(root, compatibility)
    changed_platform = deploy_host_bundle.dependency_identity(
        root, {**compatibility, "architecture": "different"}
    )

    assert first != second
    assert second != changed_platform


def test_deploy_host_bundle_rejects_tampered_payload(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path)
    archive = tmp_path / "bundle.tar.gz"
    deploy_host_bundle.write_deterministic_archive(root, archive)
    extracted = deploy_host_bundle.extract_verified_bundle(
        archive, tmp_path / "extracted"
    )
    (extracted / "requirements/deploy-host.lock").write_text(
        "tampered\n", encoding="utf-8"
    )

    with pytest.raises(
        deploy_host_bundle.DeployHostBundleError, match="digest mismatch"
    ):
        deploy_host_bundle.verify_bundle_tree(extracted)


def test_deploy_host_bundle_rejects_incompatible_platform(tmp_path: Path) -> None:
    root = _bundle_tree(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["compatibility"]["architecture"] = "different-architecture"

    with pytest.raises(
        deploy_host_bundle.DeployHostBundleError, match="incompatible with this host"
    ):
        deploy_host_bundle.validate_host_compatibility(manifest)


def test_deploy_host_payload_identity_allows_cross_commit_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = iter(("", json.dumps({"sha256": "a" * 64})))
    monkeypatch.setattr(
        setup_deploy_host, "_run", lambda *_args, **_kwargs: next(outputs)
    )

    report = setup_deploy_host.validate_bundle_source(
        {"source": {"payload_identity_sha256": "a" * 64}},
        repo_root=tmp_path,
        allow_source_mismatch=False,
    )

    assert report["matched"] is True
    assert report["expected_payload_identity_sha256"] == "a" * 64


def test_setup_requires_signed_bundle_or_explicit_network(tmp_path: Path) -> None:
    with pytest.raises(setup_deploy_host.DeployHostSetupError, match="signed --bundle"):
        setup_deploy_host.setup_deploy_host(
            repo_root=tmp_path,
            venv=tmp_path / ".venv",
            python="python3.12",
            archive=None,
            signature_bundle=None,
            cosign_key=None,
            certificate_identity=None,
            certificate_oidc_issuer=None,
            allow_unsigned=False,
            allow_network=False,
            allow_source_mismatch=False,
        )


def test_bundle_setup_reinstalls_project_wheel_without_source_path_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every pip call runs with the ambient Python path stripped.

    A deployment host that exports PYTHONPATH (a checked-out repository, say)
    would otherwise let the admin CLI import that tree instead of the wheel the
    signed bundle installed, so the installed digest would no longer describe
    the code being run. ``--force-reinstall`` is what makes the wheel win over
    an equal-version install already in the venv.
    """

    bundle = tmp_path / "bundle"
    wheelhouse = bundle / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    (bundle / "build.lock").write_text("", encoding="utf-8")
    (bundle / "deploy-host.lock").write_text("", encoding="utf-8")
    (bundle / "project.whl").write_text("", encoding="utf-8")
    (bundle / "admin-config.example.yaml").write_text(
        "kind: AdminConfig\n", encoding="utf-8"
    )
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "requirements": {
                    "build": "build.lock",
                    "deploy_host": "deploy-host.lock",
                },
                "project_wheel": "project.whl",
                "admin_config_template": "admin-config.example.yaml",
            }
        ),
        encoding="utf-8",
    )
    commands: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "src"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "python"))
    monkeypatch.setattr(
        setup_deploy_host,
        "_run",
        lambda command, *, env=None, **_kwargs: commands.append(
            ([str(item) for item in command], dict(env or {}))
        )
        or "",
    )
    monkeypatch.setattr(
        setup_deploy_host, "_install_bundled_tools", lambda *_a, **_k: None
    )
    venv = tmp_path / "venv"

    setup_deploy_host.install_from_bundle(bundle, venv=venv)

    assert commands, "the bundle install ran no commands"
    for command, environment in commands:
        assert "PYTHONPATH" not in environment, command
        assert "PYTHONHOME" not in environment, command
    project_install = commands[-1][0]
    assert "--force-reinstall" in project_install
    assert project_install[-1] == str(bundle / "project.whl")
    installed = venv / "share/gpu-fault/admin-config.example.yaml"
    assert installed.read_text(encoding="utf-8") == "kind: AdminConfig\n"


def test_incomplete_venv_state_is_reported_as_incomplete(tmp_path: Path) -> None:
    """The reuse path keys on this exact message to fall back to a reinstall.

    A venv whose recorded state has a distribution but no module digest cannot
    be identity-checked. That is a reinstall, not a hard failure, so the message
    has to stay recognisable to the caller that swallows it.
    """

    with pytest.raises(
        setup_deploy_host.DeployHostSetupError,
        match="deployment-host venv is incomplete",
    ):
        setup_deploy_host.deploy_host_project_report(
            tmp_path / "venv", {"project_distribution": "gpu-fault-control-plane"}
        )


def test_setup_installs_admin_config_template_read_only(tmp_path: Path) -> None:
    source = tmp_path / "admin-config.example.yaml"
    source.write_text("kind: AdminConfig\n", encoding="utf-8")
    venv = tmp_path / "venv"

    installed = setup_deploy_host.install_admin_config_template(source, venv)

    assert installed == venv / "share/gpu-fault/admin-config.example.yaml"
    assert installed.read_text(encoding="utf-8") == "kind: AdminConfig\n"
    assert installed.stat().st_mode & 0o777 == 0o644


def test_layered_bundle_install_reuses_dependency_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "requirements").mkdir(parents=True)
    (bundle / "wheelhouse").mkdir()
    (bundle / "config").mkdir()
    for name in ("build.lock", "deploy-host.lock"):
        (bundle / "requirements" / name).write_text("", encoding="utf-8")
    project_wheel = bundle / "wheelhouse/project.whl"
    project_wheel.write_bytes(b"wheel")
    template = bundle / "config/admin-config.example.yaml"
    template.write_text("kind: AdminConfig\n", encoding="utf-8")
    manifest = {
        "requirements": {
            "build": "requirements/build.lock",
            "deploy_host": "requirements/deploy-host.lock",
        },
        "project_wheel": "wheelhouse/project.whl",
        "admin_config_template": "config/admin-config.example.yaml",
        "dependency_identity_sha256": "a" * 64,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    venv = tmp_path / "overlay"
    dependency_venv = tmp_path / "dependencies"
    (venv / "bin").mkdir(parents=True)
    (dependency_venv / "bin").mkdir(parents=True)
    ruff = dependency_venv / "bin/ruff"
    ruff.write_text("#!/bin/sh\n", encoding="utf-8")
    ruff.chmod(0o755)
    commands = []
    monkeypatch.setattr(
        setup_deploy_host,
        "_venv_site_paths",
        lambda path: (
            (tmp_path / "overlay-site")
            if path == venv
            else (tmp_path / "dependency-site"),
        ),
    )
    monkeypatch.setattr(
        setup_deploy_host,
        "_pip_install",
        lambda *_args, **_kwargs: pytest.fail(
            "layered overlay reinstalled dependency locks"
        ),
    )
    monkeypatch.setattr(
        setup_deploy_host,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)) or "",
    )

    setup_deploy_host.install_from_bundle(
        bundle, venv=venv, dependency_venv=dependency_venv
    )

    pth = tmp_path / "overlay-site/gpu-fault-deploy-host-dependencies.pth"
    assert pth.read_text(encoding="utf-8") == f"{tmp_path / 'dependency-site'}\n"
    assert len(commands) == 1
    assert str(project_wheel) in commands[0]
    assert (venv / "bin/ruff").resolve() == ruff.resolve()


def test_existing_overlay_revalidates_dependency_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv = tmp_path / "venv"
    dependency_venv = tmp_path / "dependencies"
    (venv / "bin").mkdir(parents=True)
    (dependency_venv / "bin").mkdir(parents=True)
    dependency_ruff = dependency_venv / "bin/ruff"
    dependency_ruff.write_text("#!/bin/sh\n", encoding="utf-8")
    dependency_ruff.chmod(0o755)
    (venv / "bin/ruff").symlink_to(dependency_ruff)
    (venv / "bin/python").write_text("", encoding="utf-8")
    (venv / "bin/gpu-fault-admin").write_text("", encoding="utf-8")
    (venv / "share/gpu-fault").mkdir(parents=True)
    (venv / "share/gpu-fault/admin-config.example.yaml").write_text(
        "kind: AdminConfig\n", encoding="utf-8"
    )
    (venv / setup_deploy_host.STATE_NAME).write_text(
        json.dumps(
            {
                "dependency_identity_sha256": "a" * 64,
                "dependency_venv": str(dependency_venv),
            }
        ),
        encoding="utf-8",
    )
    checked: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        setup_deploy_host,
        "_check_dependency_venv",
        lambda path, identity: checked.append((path, identity)),
    )
    monkeypatch.setattr(setup_deploy_host, "_run", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        setup_deploy_host,
        "_dependency_report",
        lambda *_args, **_kwargs: {"healthy": True},
    )

    setup_deploy_host.check_deploy_host_venv(venv)

    assert checked == [(dependency_venv, "a" * 64)]


def test_deploy_host_project_identity_matches_separate_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv = tmp_path / "venv"
    expected = "a" * 64
    environments = []
    monkeypatch.setenv("PYTHONPATH", "/leaked/source")
    monkeypatch.setattr(
        setup_deploy_host,
        "_run",
        lambda *_args, **kwargs: (
            environments.append(kwargs.get("env"))
            or json.dumps(
                {
                    "distribution": "gpu-fault-deploy-host",
                    "module_digest": expected,
                    "version": "0.10.0",
                }
            )
        ),
    )

    report = setup_deploy_host.deploy_host_project_report(
        venv,
        {
            "project_distribution": "gpu-fault-deploy-host",
            "project_module_digest": expected,
        },
    )

    assert report == {
        "distribution": "gpu-fault-deploy-host",
        "module_digest": expected,
        "version": "0.10.0",
    }
    assert "PYTHONPATH" not in environments[0]


def test_venv_activation_can_restore_previous_symlink(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    target = tmp_path / "venv"
    target.symlink_to(old, target_is_directory=True)

    backup = setup_deploy_host.activate_venv(new, target)

    assert target.resolve() == new
    assert backup is not None and backup.is_symlink()
    setup_deploy_host.restore_venv_activation(target, backup)
    assert target.resolve() == old

    backup = setup_deploy_host.activate_venv(new, target)
    setup_deploy_host.finalize_venv_activation(backup)
    assert target.resolve() == new
    assert backup is not None and not backup.exists()


def test_deploy_host_project_identity_rejects_runtime_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        setup_deploy_host,
        "_run",
        lambda *_args, **_kwargs: json.dumps(
            {
                "distribution": "gpu-fault-deploy-host",
                "module_digest": "b" * 64,
                "version": "0.10.0",
            }
        ),
    )

    with pytest.raises(
        setup_deploy_host.DeployHostSetupError, match="project identity does not match"
    ):
        setup_deploy_host.deploy_host_project_report(
            tmp_path / "venv",
            {
                "project_distribution": "gpu-fault-deploy-host",
                "project_module_digest": "a" * 64,
            },
        )


def test_bundle_signature_verification_uses_cosign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "bundle.tar.gz"
    signature = tmp_path / "bundle.sigstore.json"
    archive.write_bytes(b"bundle")
    signature.write_text("{}", encoding="utf-8")
    commands = []
    monkeypatch.setattr(
        setup_deploy_host,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)) or "",
    )

    setup_deploy_host.verify_bundle_signature(
        archive,
        signature_bundle=signature,
        cosign_key="/secure/cosign.pub",
        certificate_identity=None,
        certificate_oidc_issuer=None,
        allow_unsigned=False,
    )

    assert commands == [
        [
            "cosign",
            "verify-blob",
            "--bundle",
            str(signature),
            "--key",
            "/secure/cosign.pub",
            str(archive),
        ]
    ]


def test_bundle_source_binding_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = iter(("current-commit", " M changed.py"))
    monkeypatch.setattr(
        setup_deploy_host, "_run", lambda *_args, **_kwargs: next(outputs)
    )

    with pytest.raises(
        setup_deploy_host.DeployHostSetupError, match="source does not match"
    ):
        setup_deploy_host.validate_bundle_source(
            {"source": {"git_commit": "expected-commit", "dirty": False}},
            repo_root=tmp_path,
            allow_source_mismatch=False,
        )


def test_dependency_report_records_versions_without_credentials() -> None:
    manifest = {
        "schema_version": 1,
        "python": {"major": 3, "minor": 12},
        "tools": [
            {
                "name": "example",
                "executable": "example",
                "command": ["example", "--version"],
            }
        ],
    }

    def run(arguments, **_kwargs):
        return subprocess.CompletedProcess(arguments, 0, "example 1.2.3\n", "")

    report = deploy_host_dependency_report(
        manifest=manifest, which=lambda _name: "/opt/tools/example", runner=run
    )

    assert report["healthy"] is True
    assert report["tools"] == [
        {
            "name": "example",
            "executable": "/opt/tools/example",
            "status": "PASS",
            "version": "example 1.2.3",
        }
    ]
