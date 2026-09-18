from __future__ import annotations

import json
import site
import subprocess
import sys
import tomllib
import venv
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tests._installed_test_dependencies import dependency_site_directories


def test_crypto_fixtures_have_an_explicit_development_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    development = {
        Requirement(value).name for value in project["optional-dependencies"]["dev"]
    }
    runtime = {Requirement(value).name for value in project["dependencies"]}
    assert "cryptography" in development, (
        "custody and TLS fixtures require cryptography"
    )
    assert "cryptography" not in runtime, (
        "test crypto must not become a runtime dependency"
    )


@pytest.mark.parametrize("shared_layer", [False, True], ids=["standalone", "overlay"])
def test_installed_fixture_dependencies_preserve_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_layer: bool
) -> None:
    parent = tmp_path / "parent/site-packages"
    shared = tmp_path / "shared/site-packages"
    source = tmp_path / "checkout/src"
    for path in (parent, shared, source):
        path.mkdir(parents=True)
    dependency = shared if shared_layer else parent
    (dependency / "fixture_dependency.py").write_text("VALUE = 'installed'\n")
    (source / "fixture_dependency.py").write_text(
        "raise AssertionError('checkout dependencies must not be imported')\n"
    )
    marker = tmp_path / "pth-executed"
    for path in (parent, shared):
        (path / "fixture.pth").write_text(
            f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n"
        )
    with monkeypatch.context() as patch:
        patch.setattr(site, "getsitepackages", lambda: [str(parent)])
        patch.setattr(
            sys,
            "path",
            [str(source), str(parent), *([str(shared)] if shared_layer else [])],
        )
        directories = dependency_site_directories()
    assert directories == (
        (str(parent), str(shared)) if shared_layer else (str(parent),)
    ), "fixture imports must retain active dependency layers, not checkout paths"
    assert not marker.exists(), "discovering directories must not execute .pth files"

    host = tmp_path / "host"
    venv.EnvBuilder(with_pip=False).create(host)
    installed = (
        host
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    (installed / "fixture-dependencies.pth").write_text("\n".join(directories) + "\n")
    result = subprocess.run(
        [
            str(host / "bin/python"),
            "-I",
            "-B",
            "-c",
            "import json, fixture_dependency; "
            "print(json.dumps([fixture_dependency.VALUE, fixture_dependency.__file__]))",
        ],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "installed",
        str(dependency / "fixture_dependency.py"),
    ], "the isolated child must import the installed dependency in either layout"
    assert not marker.exists(), "the child must not replay another environment's .pth"


def test_dependency_directories_ignore_unusable_paths_and_deduplicate_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = tmp_path / "installed/site-packages"
    installed.mkdir(parents=True)
    alias = tmp_path / "alias/site-packages"
    alias.parent.mkdir()
    alias.symlink_to(installed, target_is_directory=True)
    distro = tmp_path / "system/dist-packages"
    distro.mkdir(parents=True)
    relative = tmp_path / "relative/site-packages"
    relative.mkdir(parents=True)
    ordinary_file = tmp_path / "file/site-packages"
    ordinary_file.parent.mkdir()
    ordinary_file.write_text("not a directory\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(installed), str(alias)])
    monkeypatch.setattr(
        sys,
        "path",
        [
            "",
            "relative/site-packages",
            str(tmp_path / "missing/site-packages"),
            str(ordinary_file),
            str(installed),
            str(distro),
        ],
    )
    assert dependency_site_directories() == (str(installed), str(distro)), (
        "only existing absolute package directories may enter the child, once each"
    )
