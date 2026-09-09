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
