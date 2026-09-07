"""The lazy ``_EXPORTS`` tables and manifest entry points must resolve.

Fifteen package ``__init__.py`` files under ``src/gpu_fault/`` publish names
through ``_EXPORTS = {"Name": ("module", "attribute")}`` and a module
``__getattr__``. Nothing executes those tuples until a caller asks for the
name, so a typo or a moved function survives every import-time check and
surfaces as ``AttributeError`` in whoever asks first. The Lambda template's
``Handler:`` is the same shape one level up: a dotted string the runtime
resolves at cold start, with no test standing between the manifest and the
package it names.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check-lazy-exports.py"
MODULE = lazy_script_module(SCRIPT)


def _write_package(root: Path, name: str, init_body: str) -> Path:
    package = root / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(textwrap.dedent(init_body), encoding="utf-8")
    (package / "inner.py").write_text("value = 1\n", encoding="utf-8")
    return package


def test_discovery_parses_the_table_without_importing_the_package(
    tmp_path: Path,
) -> None:
    _write_package(
        tmp_path / "src",
        "lazygate_discover",
        """
        raise RuntimeError("importing this package must not happen")
        _EXPORTS = {
            "value": ("lazygate_discover.inner", "value"),
        }
        """,
    )

    tables = MODULE.discover_export_tables(tmp_path / "src")

    assert tables == {
        "lazygate_discover": {"value": ("lazygate_discover.inner", "value")}
    }


def test_unresolvable_tuples_are_all_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_package(
        tmp_path / "src",
        "lazygate_broken",
        """
        _EXPORTS = {
            "good": ("lazygate_broken.inner", "value"),
            "missing_attribute": ("lazygate_broken.inner", "absent"),
            "missing_module": ("lazygate_broken.nowhere", "value"),
        }
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path / "src"))

    failures = MODULE.lazy_export_failures(tmp_path / "src")

    assert [failure.split(":", 1)[0] for failure in failures] == [
        "lazygate_broken.missing_attribute",
        "lazygate_broken.missing_module",
    ]
    assert "absent" in failures[0]
    assert "lazygate_broken.nowhere" in failures[1]


def test_cloudformation_extractor_reads_python_lambda_handlers(tmp_path: Path) -> None:
    template = tmp_path / "template.yaml"
    template.write_text(
        textwrap.dedent(
            """
            Resources:
              Fn:
                Type: AWS::Lambda::Function
                Properties:
                  Runtime: python3.12
                  Handler: lazygate_pkg.handler
                  Role: !GetAtt Role.Arn
              NodeFn:
                Type: AWS::Lambda::Function
                Properties:
                  Runtime: nodejs20.x
                  Handler: index.handler
              Queue:
                Type: AWS::SQS::Queue
            """
        ),
        encoding="utf-8",
    )

    assert MODULE.cloudformation_python_handlers(template) == ["lazygate_pkg.handler"]


def test_manifest_scan_reports_a_handler_the_package_does_not_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_package(
        tmp_path / "src",
        "lazygate_lambda",
        """
        _EXPORTS = {"present": ("lazygate_lambda.inner", "value")}
        __all__ = list(_EXPORTS)

        def __getattr__(name):
            module, attribute = _EXPORTS[name]
            from importlib import import_module
            return getattr(import_module(module), attribute)
        """,
    )
    manifests = tmp_path / "deploy/aws/lambda"
    manifests.mkdir(parents=True)
    (manifests / "a.yaml").write_text(
        textwrap.dedent(
            """
            Resources:
              Fn:
                Type: AWS::Lambda::Function
                Properties:
                  Runtime: python3.12
                  Handler: lazygate_lambda.missing_handler
              Ok:
                Type: AWS::Lambda::Function
                Properties:
                  Runtime: python3.12
                  Handler: lazygate_lambda.present
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path / "src"))

    failures = MODULE.manifest_entry_point_failures(tmp_path)

    assert len(failures) == 1
    assert "deploy/aws/lambda/a.yaml" in failures[0]
    assert "lazygate_lambda.missing_handler" in failures[0]


def test_cli_fails_on_a_broken_synthetic_root(tmp_path: Path) -> None:
    _write_package(
        tmp_path / "src",
        "lazygate_cli",
        """
        _EXPORTS = {"broken": ("lazygate_cli.inner", "absent")}
        """,
    )
    (tmp_path / "deploy/aws/lambda").mkdir(parents=True)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "lazygate_cli.broken" in result.stdout


def test_repository_exports_and_manifest_entry_points_resolve() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
