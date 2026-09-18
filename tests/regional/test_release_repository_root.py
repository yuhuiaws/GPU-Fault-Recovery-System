"""``gpu_fault_release.repository_root``: where an installed engine finds ``deploy/``.

The deploy-host wheel carries the release engine so the admin CLI's module-level
imports resolve at the wheel's own version; installed under site-packages the
old ``Path(__file__).parents[2]`` anchor pointed nowhere and the first deploy
from main after aca7a37 died importing ``regional_deployment_inventory``
(2026-09-09). Running from a checkout nothing changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import gpu_fault_release
from gpu_fault_release import (
    REPOSITORY_ROOT_ENV,
    STATE_DIR_BINDING,
    resolve_repository_root,
)

ROOT = Path(__file__).resolve().parents[2]


def _fake_repository(path: Path) -> Path:
    (path / "deploy/control-plane/regional").mkdir(parents=True)
    (path / "scripts").mkdir()
    return path


def test_a_checkout_resolves_to_itself() -> None:
    package_file = ROOT / "src/gpu_fault_release/__init__.py"
    assert resolve_repository_root(package_file, environ={}) == ROOT, (
        "a checkout is its own root"
    )
    assert gpu_fault_release.repository_root() == ROOT, (
        "the cached root is the checkout"
    )


def test_an_installed_copy_reads_the_declared_root(tmp_path: Path) -> None:
    installed = (
        tmp_path / "venv/lib/python3.12/site-packages/gpu_fault_release/__init__.py"
    )
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    repository = _fake_repository(tmp_path / "snapshot")

    resolved = resolve_repository_root(
        installed,
        environ={REPOSITORY_ROOT_ENV: str(repository)},
        prefix=tmp_path / "none",
    )

    assert resolved == repository.resolve(), resolved


def test_an_installed_copy_follows_the_venv_binding_to_the_site(tmp_path: Path) -> None:
    prefix = tmp_path / "venv"
    installed = prefix / "lib/python3.12/site-packages/gpu_fault_release/__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    repository = _fake_repository(tmp_path / "snapshot")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text(
        "apiVersion: gpu-fault.io/v1\nspec:\n  repositoryRoot: "
        f"{repository}\n  other: x\n",
        encoding="utf-8",
    )
    (prefix / STATE_DIR_BINDING).write_text(
        json.dumps({"schema_version": 1, "state_dir": str(state_dir)}), encoding="utf-8"
    )

    resolved = resolve_repository_root(installed, environ={}, prefix=prefix)

    assert resolved == repository.resolve(), resolved


def test_an_unlocatable_root_names_every_attempt(tmp_path: Path) -> None:
    installed = (
        tmp_path / "venv/lib/python3.12/site-packages/gpu_fault_release/__init__.py"
    )
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match=REPOSITORY_ROOT_ENV) as raised:
        resolve_repository_root(
            installed,
            environ={REPOSITORY_ROOT_ENV: str(tmp_path / "missing")},
            prefix=tmp_path / "venv",
        )
    assert STATE_DIR_BINDING in str(raised.value), raised.value


@pytest.fixture
def installed_binding(tmp_path: Path) -> tuple[Path, Path, Path]:
    prefix = tmp_path / "venv"
    installed = prefix / "lib/python3.12/site-packages/gpu_fault_release/__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (prefix / STATE_DIR_BINDING).write_text(
        json.dumps({"schema_version": 1, "state_dir": str(state_dir)}), encoding="utf-8"
    )
    return prefix, installed, state_dir / "site.yaml"


@pytest.mark.parametrize("style", [None, "'", '"', "|", ">"])
def test_binding_reads_yaml_scalar_styles_and_folded_paths(
    installed_binding: tuple[Path, Path, Path], style: str | None
) -> None:
    prefix, installed, site = installed_binding
    repository = _fake_repository(
        site.parent / ("release snapshot with spaces " * 4).strip() / "checkout"
    )
    site.write_text(
        yaml.safe_dump(
            {"spec": {"repositoryRoot": str(repository)}}, default_style=style
        ),
        encoding="utf-8",
    )

    assert (
        resolve_repository_root(installed, environ={}, prefix=prefix)
        == repository.resolve()
    )


def test_binding_reads_quoted_paths_and_ignores_yaml_comments(
    installed_binding: tuple[Path, Path, Path],
) -> None:
    prefix, installed, site = installed_binding
    repository = _fake_repository(site.parent / "snapshot 'one' \"two\"")
    site.write_text(
        "spec:\n"
        f"  repositoryRoot: {json.dumps(str(repository))} # release input path\n",
        encoding="utf-8",
    )

    assert (
        resolve_repository_root(installed, environ={}, prefix=prefix)
        == repository.resolve()
    )


@pytest.mark.parametrize("relative", ["snapshot", "../snapshot"])
def test_binding_resolves_paths_relative_to_the_site_not_the_process(
    installed_binding: tuple[Path, Path, Path],
    relative: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefix, installed, site = installed_binding
    repository = _fake_repository((site.parent / relative).resolve())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    site.write_text(
        yaml.safe_dump({"spec": {"repositoryRoot": relative}}), encoding="utf-8"
    )

    assert (
        resolve_repository_root(installed, environ={}, prefix=prefix)
        == repository.resolve()
    )


def test_binding_expands_a_home_relative_path(
    installed_binding: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, installed, site = installed_binding
    home = site.parent / "home"
    repository = _fake_repository(home / "snapshot")
    monkeypatch.setenv("HOME", str(home))
    site.write_text("spec:\n  repositoryRoot: ~/snapshot\n", encoding="utf-8")

    assert (
        resolve_repository_root(installed, environ={}, prefix=prefix)
        == repository.resolve()
    )


@pytest.mark.parametrize(
    "document",
    [
        None,
        [],
        "not a site",
        {},
        {"repositoryRoot": "."},
        {"spec": None},
        {"spec": []},
        {"spec": "repositoryRoot: ."},
        {"spec": {}},
        {"spec": {"repositoryRoot": None}},
        {"spec": {"repositoryRoot": False}},
        {"spec": {"repositoryRoot": 17}},
        {"spec": {"repositoryRoot": []}},
        {"spec": {"repositoryRoot": {}}},
        {"spec": {"repositoryRoot": ""}},
        {"spec": {"repositoryRoot": " \t "}},
    ],
)
def test_binding_rejects_invalid_document_spec_and_root_types(
    installed_binding: tuple[Path, Path, Path], document: object
) -> None:
    prefix, installed, site = installed_binding
    site.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot locate its repository root"):
        resolve_repository_root(installed, environ={}, prefix=prefix)


@pytest.mark.parametrize(
    "text",
    [
        "spec: [unterminated",
        "spec:\n  repositoryRoot: !!python/object:invalid {}\n",
        "spec: {}\n---\nspec: {}\n",
    ],
)
def test_binding_rejects_malformed_or_unsafe_yaml(
    installed_binding: tuple[Path, Path, Path], text: str
) -> None:
    prefix, installed, site = installed_binding
    site.write_text(text, encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot locate its repository root"):
        resolve_repository_root(installed, environ={}, prefix=prefix)


def test_binding_rejects_an_unreadable_site(
    installed_binding: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, installed, site = installed_binding
    site.write_text("spec: {}\n", encoding="utf-8")
    read_text = Path.read_text

    def read(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if path == site:
            raise PermissionError("fixture site is unreadable")
        return read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(RuntimeError, match="cannot locate its repository root"):
        resolve_repository_root(installed, environ={}, prefix=prefix)


@pytest.mark.parametrize("valid_override", [False, True])
def test_override_precedence_still_requires_repository_anchors(
    installed_binding: tuple[Path, Path, Path], valid_override: bool
) -> None:
    prefix, installed, site = installed_binding
    bound = _fake_repository(site.parent / "bound")
    override = site.parent / "override"
    override.mkdir()
    if valid_override:
        _fake_repository(override)
    site.write_text(
        yaml.safe_dump({"spec": {"repositoryRoot": str(bound)}}), encoding="utf-8"
    )

    resolved = resolve_repository_root(
        installed, environ={REPOSITORY_ROOT_ENV: str(override)}, prefix=prefix
    )

    assert resolved == (override if valid_override else bound).resolve()


@pytest.mark.parametrize("missing", ["scripts", "deploy/control-plane/regional"])
def test_bound_repository_requires_both_anchors(
    installed_binding: tuple[Path, Path, Path], missing: str
) -> None:
    prefix, installed, site = installed_binding
    repository = _fake_repository(site.parent / "snapshot")
    (repository / missing).rmdir()
    site.write_text(
        yaml.safe_dump({"spec": {"repositoryRoot": str(repository)}}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="cannot locate its repository root"):
        resolve_repository_root(installed, environ={}, prefix=prefix)
