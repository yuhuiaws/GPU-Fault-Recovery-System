"""Installed observability uses the engine locator, never site-packages assets."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

import gpu_fault_release
from gpu_fault_release import regional_dataplane_observability as observability


def _repository(path: Path) -> Path:
    for directory in (
        "deploy/control-plane/regional",
        "scripts",
        "deploy/dataplane",
        "deploy/observability",
    ):
        (path / directory).mkdir(parents=True)
    (path / "deploy/observability/install-amp-monitoring.sh").write_text(
        "not an executable installer\n", encoding="utf-8"
    )
    (path / observability.DATAPLANE_ADOT_MANIFEST).write_text(
        yaml.safe_dump(
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "snapshot-adot"},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def installed_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    prefix = tmp_path / "venv"
    installed = (
        prefix
        / "lib/python3.12/site-packages/gpu_fault_release"
        / "regional_dataplane_observability.py"
    )
    installed.parent.mkdir(parents=True)
    shutil.copyfile(observability.__file__, installed)
    repository = _repository(tmp_path / "snapshot with spaces")
    state = tmp_path / "state"
    state.mkdir()
    (state / "site.yaml").write_text(
        yaml.safe_dump({"spec": {"repositoryRoot": "../snapshot with spaces"}}),
        encoding="utf-8",
    )
    binding = prefix / gpu_fault_release.STATE_DIR_BINDING
    binding.write_text(
        json.dumps({"schema_version": 1, "state_dir": str(state)}), encoding="utf-8"
    )
    binding.chmod(0o600)
    return prefix, installed, repository


def _load_installed(
    installed: Path, locator: Callable[[], Path], monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    # Isolate module initialization without reloading shared engine imports.
    name = "gpu_fault_release._installed_observability_test"
    spec = importlib.util.spec_from_file_location(name, installed)
    assert spec is not None and spec.loader is not None, "cannot load fixture module"
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as local:
        local.setattr(gpu_fault_release, "repository_root", locator)
        local.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("root_source", ["binding", "override"])
def test_installed_observability_resolves_both_assets_through_the_engine(
    installed_assets: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    root_source: str,
    tmp_path: Path,
) -> None:
    prefix, installed, repository = installed_assets
    environment: dict[str, str] = {}
    if root_source == "override":
        repository = _repository(tmp_path / "override")
        environment[gpu_fault_release.REPOSITORY_ROOT_ENV] = str(repository)
    lookups: list[Path] = []

    def locate() -> Path:
        result = gpu_fault_release.resolve_repository_root(
            installed, environ=environment, prefix=prefix
        )
        lookups.append(result)
        return result

    module = _load_installed(installed, locate, monkeypatch)
    assert lookups == [repository], "observability bypassed the engine locator"
    assert module.ROOT == repository, "observability resolved a different snapshot"
    assert [item["name"] for item in module.declared_dataplane_adot_objects()] == [
        "snapshot-adot"
    ], "the ADOT manifest did not come from the selected snapshot"
    calls: list[tuple[list[str], dict[str, str]]] = []

    def record(arguments: list[str], *, env: dict[str, str]) -> None:
        calls.append((arguments, env))

    release = SimpleNamespace(runner=SimpleNamespace(run=record))
    module.run_amp_monitoring_installer(
        release, {"RULE_NAMESPACE": "fixture-rules"}, include_expected_rules=False
    )
    assert calls == [
        (
            [
                "bash",
                str(repository / "deploy/observability/install-amp-monitoring.sh"),
                "--runtime-only",
            ],
            {"RULE_NAMESPACE": "fixture-rules"},
        )
    ], "the fake runner received an installer outside the selected snapshot"


def test_installed_observability_propagates_an_unlocatable_engine_root(
    installed_assets: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, installed, _repository_path = installed_assets

    def locate() -> Path:
        return gpu_fault_release.resolve_repository_root(
            installed, environ={}, prefix=prefix / "unbound"
        )

    with pytest.raises(RuntimeError, match="cannot locate its repository root"):
        _load_installed(installed, locate, monkeypatch)


def test_missing_snapshot_manifest_never_falls_back_to_site_packages(
    installed_assets: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, installed, repository = installed_assets
    (repository / observability.DATAPLANE_ADOT_MANIFEST).unlink()
    decoy = installed.parents[2] / observability.DATAPLANE_ADOT_MANIFEST
    decoy.parent.mkdir(parents=True)
    decoy.write_text(
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: wrong-adot\n",
        encoding="utf-8",
    )
    module = _load_installed(
        installed,
        lambda: gpu_fault_release.resolve_repository_root(
            installed, environ={}, prefix=prefix
        ),
        monkeypatch,
    )
    with pytest.raises(FileNotFoundError) as raised:
        module.declared_dataplane_adot_objects()
    assert raised.value.filename == str(
        repository / observability.DATAPLANE_ADOT_MANIFEST
    ), "missing snapshot assets must fail without a fallback"
