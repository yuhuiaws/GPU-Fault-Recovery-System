"""An uninstall resumes with foreign cleanup tools only explicitly and journaled."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault.admin import uninstall_override as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault.admin.uninstall_types import UninstallRequest
from tests.admin.test_admin_site import site_file


def _foreign_tree(tmp_path: Path) -> Path:
    root = tmp_path / "foreign"
    rollout = root / "deploy/control-plane/regional/rollout-regional-release.sh"
    rollout.parent.mkdir(parents=True)
    rollout.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    rollout.chmod(0o755)
    (root / "deploy/control-plane/tools").mkdir()
    (root / "deploy/control-plane/tools/cleanup_kubernetes.py").write_text(
        "print('fixed')\n", encoding="utf-8"
    )
    (root / "scripts").mkdir()
    (root / "scripts/tool.sh").write_text("exit 0\n", encoding="utf-8")
    (root / "deploy/control-plane/tools/__pycache__").mkdir()
    (root / "deploy/control-plane/tools/__pycache__/x.pyc").write_bytes(b"\x00")
    return root


def _request(tmp_path: Path, *, override: bool, foreign: bool) -> UninstallRequest:
    site_path = site_file(tmp_path)
    root = _foreign_tree(tmp_path) if foreign else None
    return UninstallRequest(
        site=load_site(site_path, repository_root=root),
        cpu_disposition="keep",
        confirmation="UNINSTALL_GPU_FAULT",
        reset_database=True,
        repository_root_override=override,
    )


def test_the_deployed_snapshot_needs_no_override_and_writes_nothing(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, override=False, foreign=False)
    state_path = tmp_path / "uninstall" / "state.json"
    state = {"phase": "REGISTRY_EXPORTED"}

    module.record_repository_root_override(request, state_path, state, resumed=True)

    assert not state_path.exists(), "the deployed snapshot leaves no override record"
    assert "repository_root_overrides" not in state, "no override to journal"


def test_a_foreign_tree_without_the_flag_is_refused(tmp_path: Path) -> None:
    request = _request(tmp_path, override=False, foreign=True)
    with pytest.raises(BootstrapError, match="accept-repository-root-override"):
        module.record_repository_root_override(
            request,
            tmp_path / "state.json",
            {"phase": "REGISTRY_EXPORTED"},
            resumed=True,
        )


@pytest.mark.parametrize(
    ("resumed", "phase"), [(False, "STARTED"), (True, "COMPLETED")]
)
def test_the_override_needs_an_uninstall_in_progress(
    tmp_path: Path, resumed: bool, phase: str
) -> None:
    request = _request(tmp_path, override=True, foreign=True)
    with pytest.raises(BootstrapError, match="already in progress"):
        module.record_repository_root_override(
            request, tmp_path / "state.json", {"phase": phase}, resumed=resumed
        )


def test_the_override_is_journaled_once_per_tool_tree(tmp_path: Path) -> None:
    request = _request(tmp_path, override=True, foreign=True)
    state_path = tmp_path / "uninstall" / "state.json"
    state_path.parent.mkdir()
    state = {"phase": "REGISTRY_EXPORTED"}

    module.record_repository_root_override(request, state_path, state, resumed=True)
    module.record_repository_root_override(request, state_path, state, resumed=True)

    written = json.loads(state_path.read_text(encoding="utf-8"))
    (record,) = written["repository_root_overrides"]
    assert record["path"] == str((tmp_path / "foreign").resolve()), "override path"
    assert record["deployed_repository_root"] == str((tmp_path / "repo").resolve()), (
        "the deployed root is kept beside the override"
    )
    assert record["phase"] == "REGISTRY_EXPORTED", "journaled at the current phase"
    assert len(record["tools_sha256"]) == 64, "a sha256 hex digest"
    assert record["tools_sha256"] == module.tools_tree_sha256(tmp_path / "foreign"), (
        "the digest covers the override tree"
    )

    (tmp_path / "foreign/deploy/control-plane/tools/cleanup_kubernetes.py").write_text(
        "print('fixed again')\n", encoding="utf-8"
    )
    module.record_repository_root_override(request, state_path, state, resumed=True)
    assert (
        len(
            json.loads(state_path.read_text(encoding="utf-8"))[
                "repository_root_overrides"
            ]
        )
        == 2
    ), "a changed tool tree is a new record"


def test_the_tools_digest_ignores_bytecode_caches(tmp_path: Path) -> None:
    root = _foreign_tree(tmp_path)
    before = module.tools_tree_sha256(root)
    (root / "deploy/control-plane/tools/__pycache__/x.pyc").write_bytes(b"\x01\x02")
    assert module.tools_tree_sha256(root) == before, "bytecode caches do not count"
    (tmp_path / "repo-without-scripts/deploy").mkdir(parents=True)
    with pytest.raises(BootstrapError, match="lacks scripts/"):
        module.tools_tree_sha256(tmp_path / "repo-without-scripts")


def test_the_cleanup_shell_learns_the_accepted_config_digest_only_under_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    monkeypatch.delenv(module.CLEANUP_ACCEPT_CONFIG_ENV, raising=False)
    config = tmp_path / "release.json"
    config.write_text('{"release": {"manifest": "/x"}}', encoding="utf-8")

    (tmp_path / "plain").mkdir()
    plain = module.cleanup_shell_environment(
        _request(tmp_path / "plain", override=False, foreign=False), config
    )
    assert module.CLEANUP_ACCEPT_CONFIG_ENV not in plain, "no override, no acceptance"

    (tmp_path / "override").mkdir()
    override = module.cleanup_shell_environment(
        _request(tmp_path / "override", override=True, foreign=True), config
    )
    assert override[module.CLEANUP_ACCEPT_CONFIG_ENV] == (
        hashlib.sha256(config.read_bytes()).hexdigest()
    )


def test_the_cleanup_tools_import_gpu_fault_from_the_site_source_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "release.json"
    config.write_text("{}", encoding="utf-8")
    request = _request(tmp_path, override=False, foreign=False)
    source = str(request.site.repository_root / "src")

    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert module.cleanup_shell_environment(request, config)["PYTHONPATH"] == source, (
        "the tools import gpu_fault from the tree they run from"
    )
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    assert module.cleanup_shell_environment(request, config)["PYTHONPATH"] == (
        f"{source}:/elsewhere"
    ), "an inherited path stays behind the site tree"
