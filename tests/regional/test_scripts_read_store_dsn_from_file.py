"""Scripts exec'd inside a control-plane Pod must read the Aurora DSN from the file.

``GPU_FAULT_STORE_URL`` is the value at Pod start and goes stale the moment HA-009
rotates the master password, while ``GPU_FAULT_STORE_URL_FILE`` is re-read on every
connect; the live HA-009 run failed only because helper scripts read the env var, so
every script under ``scripts/e2e/regional`` and ``scripts/perf`` is held to
``store_dsn()``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.perf.regional_capacity_registry import STORE_DSN_SNIPPET

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRS = (ROOT / "scripts" / "e2e" / "regional", ROOT / "scripts" / "perf")
DIRECT_READ = re.compile(
    r"""(environ\[|environ\.get\(|getenv\()\s*["']GPU_FAULT_STORE_URL["']"""
)
BODY_MARKERS = (
    "GPU_FAULT_STORE_URL_FILE",
    "/etc/gpu-fault/aurora/postgres-url",
    "OSError",
)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _scan(path: Path) -> tuple[list[str], list[str]]:
    """Return (env reads outside a store_dsn body, store_dsn bodies missing the file read)."""
    outside: list[str] = []
    incomplete: list[str] = []
    def_indent: int | None = None
    body: list[str] = []
    body_start = 0

    def close_body() -> None:
        if def_indent is not None and not all(
            any(marker in line for line in body) for marker in BODY_MARKERS
        ):
            incomplete.append(f"{path.relative_to(ROOT)}:{body_start}")

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if def_indent is not None and line.strip() and _indent(line) <= def_indent:
            close_body()
            def_indent = None
        if "def store_dsn()" in line:
            def_indent = _indent(line)
            body = []
            body_start = number
            continue
        if def_indent is not None:
            body.append(line)
        elif DIRECT_READ.search(line):
            outside.append(f"{path.relative_to(ROOT)}:{number}")
    close_body()
    return outside, incomplete


def test_every_env_read_sits_inside_a_store_dsn_body() -> None:
    """A direct env read anywhere else would break again on the next rotation."""
    outside: list[str] = []
    incomplete: list[str] = []
    files = sorted(path for root in SCRIPT_DIRS for path in root.rglob("*.py"))
    assert files, f"no scripts found under {SCRIPT_DIRS}"
    for path in files:
        found, missing = _scan(path)
        outside.extend(found)
        incomplete.extend(missing)
    assert not outside, (
        "GPU_FAULT_STORE_URL read outside store_dsn(); prepend STORE_DSN_SNIPPET or "
        "copy its body: " + ", ".join(outside)
    )
    assert not incomplete, (
        "store_dsn() copies that do not read GPU_FAULT_STORE_URL_FILE first: "
        + ", ".join(incomplete)
    )


def _snippet_store_dsn():
    namespace: dict[str, object] = {}
    exec(STORE_DSN_SNIPPET, namespace)
    return namespace["store_dsn"]


def test_snippet_reads_the_mounted_file_stripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is the rotation-following source, so it wins and its newline goes."""
    dsn_file = tmp_path / "postgres-url"
    dsn_file.write_text("postgresql://file-user@example/db\n", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(dsn_file))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://stale@example/db")
    assert _snippet_store_dsn()() == "postgresql://file-user@example/db", (
        "store_dsn() must return the mounted file content without its trailing newline"
    )


def test_snippet_falls_back_to_the_env_var_without_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deploy host mounts no file, so the env var stays the fallback there."""
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://env-user@example/db")
    assert _snippet_store_dsn()() == "postgresql://env-user@example/db", (
        "store_dsn() must fall back to GPU_FAULT_STORE_URL when the file is absent"
    )
