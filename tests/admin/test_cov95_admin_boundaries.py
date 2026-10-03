from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from gpu_fault.admin import bootstrap_dependencies as dependencies
from gpu_fault.admin import deploy_consent, release_state, source_deploy
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.execution import ProofCache
from gpu_fault.admin.site import SiteConfigError, load_site
from tests.admin.test_admin_site import site_file


@pytest.fixture
def toolchain(tmp_path, monkeypatch):
    manifest = {
        "schema_version": 1,
        "python": {"major": 3, "minor": 12},
        "tools": [
            {
                "name": "fixture",
                "executable": "fixture",
                "command": ["fixture", "--version"],
            }
        ],
    }
    root = tmp_path / "package"
    data = root / dependencies.TOOL_MANIFEST
    data.parent.mkdir(parents=True)
    data.write_text(json.dumps(manifest))
    binary = tmp_path / "fixture"
    binary.write_text("example tool bytes")
    monkeypatch.setattr(dependencies, "files", lambda _package: root)
    monkeypatch.setattr(dependencies, "resolve_tool", lambda _name: str(binary))
    calls = []

    def command(arguments, **options):
        calls.append((arguments, options))
        return subprocess.CompletedProcess(arguments, 0, "fixture-version\n", "")

    monkeypatch.setattr(dependencies, "run_command", command)
    # The process-wide proof cache keys on the manifest, the tool bytes and the
    # interpreter; two tests whose tmp_path is the same path (pytest reuses
    # numbers once passed tests' directories are removed) would otherwise share
    # one proof and only the first would run the tool.
    monkeypatch.setattr(dependencies, "PROOFS", ProofCache())
    return manifest, data, calls


@pytest.mark.parametrize(
    "document",
    [[], {}, {"schema_version": 2}, {"schema_version": 1, "python": [], "tools": []}],
)
def test_tool_manifest_requires_supported_complete_shape(toolchain, document):
    toolchain[1].write_text(json.dumps(document))
    with pytest.raises(BootstrapError, match="schema_version|incomplete"):
        dependencies.load_deploy_host_tool_manifest()
    assert toolchain[2] == []


@pytest.mark.parametrize(
    "tool",
    [
        None,
        {},
        {"name": "fixture", "executable": "fixture", "command": []},
        {"name": "fixture", "executable": "fixture", "command": [None]},
    ],
)
def test_invalid_tool_definition_never_starts_a_process(toolchain, tool):
    manifest, _data, calls = toolchain
    manifest["tools"] = [tool]
    with pytest.raises(BootstrapError, match="non-object|invalid tool"):
        dependencies.deploy_host_dependency_report(manifest=manifest)
    assert calls == []


@pytest.mark.parametrize("output", ["json", "text"])
def test_dependency_main_reports_verified_fake_toolchain(toolchain, capsys, output):
    assert dependencies.main(["--output", output]) == 0
    emitted = capsys.readouterr().out
    if output == "json":
        report = json.loads(emitted)
        assert report["healthy"] is True
        assert report["tools"][0]["version"] == "fixture-version"
    else:
        assert "dependencies: PASS" in emitted
        assert "fixture: fixture-version" in emitted
    assert len(toolchain[2]) == 1


@pytest.mark.parametrize("failure", ["tool", "python"])
def test_dependency_main_refuses_missing_prerequisite(
    toolchain, monkeypatch, capsys, failure
):
    manifest, path, _calls = toolchain
    if failure == "tool":
        monkeypatch.setattr(dependencies, "resolve_tool", lambda _name: None)
    else:
        manifest["python"] = {"major": 2, "minor": 7}
        path.write_text(json.dumps(manifest))
    assert dependencies.main([]) == 2
    assert "dependency check failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "document",
    [
        '{"client":{"version":"v1"}}',
        '{"client":[]}',
        '{"client":{"version":null}}',
        "invalid",
    ],
)
def test_version_json_uses_declared_field_or_bounded_fallback(toolchain, document):
    manifest = toolchain[0]
    manifest["tools"][0]["version_json_field"] = "client.version"
    calls = []

    def runner(arguments, **options):
        calls.append(options)
        return subprocess.CompletedProcess(arguments, 0, document, "fallback")

    report = dependencies.deploy_host_dependency_report(
        manifest=manifest, runner=runner
    )
    assert report["tools"][0]["version"] == (
        "v1" if document == '{"client":{"version":"v1"}}' else document
    )
    assert calls[0]["timeout"] == 15


def repository(tmp_path):
    root = tmp_path / "repository"
    for name in source_deploy.REPOSITORY_MARKERS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "deploy":
            path.mkdir()
        else:
            path.write_text("example")
    return root


@pytest.mark.parametrize(
    "document",
    [
        "invalid",
        [],
        {"schema_version": 2},
        {"schema_version": 1},
        {"schema_version": 1, "source_repository_root": 1},
    ],
)
def test_source_repository_state_is_validated_before_fallback(tmp_path, document):
    root = repository(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / source_deploy.SOURCE_DEPLOY_STATE).write_text(
        document if isinstance(document, str) else json.dumps(document)
    )
    with pytest.raises(BootstrapError, match="invalid|no repository"):
        source_deploy.resolve_source_repository(state, current_directory=root)


@pytest.mark.parametrize(
    "document",
    [
        "invalid: [yaml",
        [],
        {},
        {"spec": {"repositoryRoot": ""}},
        {"spec": {"repositoryRoot": 1}},
    ],
)
def test_generated_site_repository_requires_valid_reference(tmp_path, document):
    (tmp_path / "site.yaml").write_text(
        document if isinstance(document, str) else yaml.safe_dump(document)
    )
    with pytest.raises(BootstrapError, match="valid repository root"):
        source_deploy.resolve_source_repository(tmp_path, current_directory=tmp_path)


def test_recorded_repository_wins_and_generated_site_is_only_fallback(tmp_path):
    root = repository(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "site.yaml").write_text(
        yaml.safe_dump({"spec": {"repositoryRoot": str(root)}})
    )
    assert (
        source_deploy.resolve_source_repository(state, current_directory=state) == root
    )
    (state / source_deploy.SOURCE_DEPLOY_STATE).write_text(
        json.dumps({"schema_version": 1, "source_repository_root": str(root)})
    )
    assert (
        source_deploy.resolve_source_repository(state, current_directory=tmp_path)
        == root
    )


def test_source_deploy_forwards_requested_targets_and_pins_pythonpath(
    tmp_path, monkeypatch
):
    root = repository(tmp_path)
    calls = []

    def driver(arguments, **options):
        calls.append((arguments, options))
        return subprocess.CompletedProcess(arguments, 7)

    monkeypatch.setattr(source_deploy, "run_driver", driver)
    status = source_deploy.run_source_deploy(
        cpu_cluster_arn="example-cpu-arn",
        gpu_cluster_arns=("example-gpu-a", "example-gpu-b"),
        state_dir=tmp_path / "state",
        admin_email="ops@example.com",
        impact_base="origin/main",
        current_directory=root,
        extra_environment={"PYTHONPATH": "untrusted", "EXAMPLE": "preserved"},
        wait_for_email_confirmation=60,
    )
    assert status == 7
    arguments, options = calls[0]
    assert arguments.count("--gpu-cluster-arn") == 2
    assert arguments[-2:] == ["--wait-for-email-confirmation", "60"]
    assert options["env"]["PYTHONPATH"] == str(root / "src")
    assert options["env"]["EXAMPLE"] == "preserved"


@pytest.mark.parametrize(
    "status,output",
    [
        (1, "{}"),
        (0, "invalid"),
        (0, "[]"),
        (0, "{}"),
        (0, '{"data":{"state.json":"invalid"}}'),
        (0, '{"data":{"state.json":"[]"}}'),
    ],
)
def test_live_release_read_errors_do_not_produce_an_identity(
    tmp_path, monkeypatch, status, output
):
    site = load_site(site_file(tmp_path))
    calls = []

    def command(arguments, **options):
        calls.append((arguments, options))
        return subprocess.CompletedProcess(arguments, status, output, "")

    monkeypatch.setattr(release_state, "run_command", command)
    with pytest.raises(SiteConfigError, match="cannot read|invalid|object"):
        release_state.live_release_state(site)
    assert len(calls) == 1 and calls[0][1]["timeout_seconds"] == 30


@pytest.mark.parametrize("content", ["invalid", "[]", '{"release_id":"example"}'])
def test_manifest_read_for_consent_is_not_an_approval(tmp_path, content):
    path = tmp_path / "manifest.json"
    assert deploy_consent.load_release_manifest(path) is None
    path.write_text(content)
    if content == "invalid":
        with pytest.raises(BootstrapError, match="invalid"):
            deploy_consent.load_release_manifest(path)
    else:
        assert deploy_consent.load_release_manifest(path) == (
            None if content == "[]" else json.loads(content)
        )
