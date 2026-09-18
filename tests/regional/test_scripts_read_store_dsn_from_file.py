"""CPU SQL probes use file-aware credentials, including their embedded scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from scripts.perf.regional_capacity_registry import STORE_DSN_SNIPPET
from tests.regional._store_dsn_contract import ENV_FILE, ENV_URL, scan_source

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRS = (ROOT / "scripts" / "e2e" / "regional", ROOT / "scripts" / "perf")
DEFAULT_FILE = "/etc/gpu-fault/aurora/postgres-url"


def _scan(path: Path) -> tuple[list[str], list[str]]:
    return scan_source(
        path.read_text(encoding="utf-8"),
        str(path.relative_to(ROOT)),
        canonical_reader=STORE_DSN_SNIPPET,
    )


def test_every_env_read_uses_a_file_aware_credential_flow() -> None:
    """A recognized flow must bind the FILE input, not merely mention its name."""
    outside: list[str] = []
    incomplete: list[str] = []
    files = sorted(path for root in SCRIPT_DIRS for path in root.rglob("*.py"))
    assert files, f"no scripts found under {SCRIPT_DIRS}"
    for path in files:
        found, missing = _scan(path)
        outside.extend(found)
        incomplete.extend(missing)
    assert not outside, (
        "startup-only or unrecognized DSN credential reads: " + ", ".join(outside)
    )
    assert not incomplete, (
        "DSN helpers differ from the tested strict reader or probes cannot parse: "
        + ", ".join(incomplete)
    )


def _snippet_store_dsn() -> Callable[[], str]:
    namespace: dict[str, Any] = {}
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


def test_explicit_missing_file_never_falls_back_to_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://env-user@example/db")
    with pytest.raises(FileNotFoundError):
        _snippet_store_dsn()()


def test_absent_default_mount_permits_the_legacy_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def missing_default(path: str, **kwargs: Any) -> None:
        calls.append((path, kwargs))
        raise FileNotFoundError

    monkeypatch.delenv(ENV_FILE, raising=False)
    monkeypatch.setenv(ENV_URL, "postgresql://legacy.invalid/db")
    monkeypatch.setattr("builtins.open", missing_default)
    assert _snippet_store_dsn()() == "postgresql://legacy.invalid/db"
    assert calls == [(DEFAULT_FILE, {"encoding": "utf-8"})]


@pytest.mark.parametrize("configured", [False, True])
def test_permission_failures_do_not_allow_a_stale_env_fallback(
    configured: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(path: str, **_kwargs: Any) -> None:
        assert path == ("/unit/projected" if configured else DEFAULT_FILE)
        raise PermissionError

    monkeypatch.delenv(ENV_FILE, raising=False)
    if configured:
        monkeypatch.setenv(ENV_FILE, "/unit/projected")
    monkeypatch.setenv(ENV_URL, "postgresql://stale.invalid/db")
    monkeypatch.setattr("builtins.open", denied)
    with pytest.raises(PermissionError):
        _snippet_store_dsn()()


def test_explicit_empty_file_path_is_not_a_default_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_FILE, "")
    monkeypatch.setenv(ENV_URL, "postgresql://stale.invalid/db")
    monkeypatch.setattr("builtins.open", lambda *_a, **_k: pytest.fail("must not open"))
    with pytest.raises(RuntimeError, match="file path is empty"):
        _snippet_store_dsn()()


@pytest.mark.parametrize("contents", ["", " \n\t"])
def test_empty_projected_file_cannot_use_the_startup_dsn(
    contents: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "projected"
    path.write_text(contents, encoding="utf-8")
    monkeypatch.setenv(ENV_FILE, str(path))
    monkeypatch.setenv(ENV_URL, "postgresql://stale.invalid/db")
    with pytest.raises(RuntimeError, match="DSN file is empty"):
        _snippet_store_dsn()()


def test_each_connection_read_follows_rotation_and_subsequent_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "projected"
    monkeypatch.setenv(ENV_FILE, str(path))
    monkeypatch.setenv(ENV_URL, "postgresql://startup.invalid/db")
    reader = _snippet_store_dsn()
    for version in ("first", "rotated"):
        path.write_text(f"postgresql://{version}.invalid/db\n", encoding="utf-8")
        assert reader() == f"postgresql://{version}.invalid/db"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        reader()


@pytest.mark.parametrize(
    ("source", "safe"),
    [
        ('dsn = os.environ["GPU_FAULT_STORE_URL"]', False),
        ('dsn = StoreCredentials(os.getenv("GPU_FAULT_STORE_URL"))', False),
        (
            'path = os.getenv("GPU_FAULT_STORE_URL_FILE")\n'
            'dsn = StoreCredentials(os.getenv("GPU_FAULT_STORE_URL"), path=path)',
            True,
        ),
        (
            'path = os.getenv("GPU_FAULT_STORE_URL_FILE")\npath = "/other"\n'
            'dsn = StoreCredentials(os.getenv("GPU_FAULT_STORE_URL"), path=path)',
            False,
        ),
        (
            'path = os.getenv("GPU_FAULT_STORE_URL_FILE")\n'
            'os.environ["GPU_FAULT_STORE_URL"] = '
            'StoreCredentials(os.environ["GPU_FAULT_STORE_URL"], path=path).conninfo()',
            True,
        ),
        (
            'path = os.environ.get("GPU_FAULT_STORE_URL_FILE", "").strip()\n'
            "url = Path(path).read_text().strip() if path else "
            'os.environ["GPU_FAULT_STORE_URL"]',
            True,
        ),
        (
            'path = os.environ.get("OTHER_PATH", "").strip()\n'
            "url = Path(path).read_text().strip() if path else "
            'os.environ["GPU_FAULT_STORE_URL"]',
            False,
        ),
        (
            'path = os.getenv("GPU_FAULT_STORE_URL_FILE")\n'
            'url = "fixed" if path else os.environ["GPU_FAULT_STORE_URL"]',
            False,
        ),
        (
            'dsn_arguments(os.environ["GPU_FAULT_STORE_URL"], expected["database"])',
            True,
        ),
        (
            'arguments = dsn_arguments(os.environ["GPU_FAULT_STORE_URL"], '
            'expected["database"])',
            False,
        ),
    ],
)
@pytest.mark.parametrize("embedded", [False, True])
def test_scanner_accepts_only_the_recognized_file_aware_flow(
    source: str, safe: bool, embedded: bool
) -> None:
    if embedded:
        source = f"PROBE = {source!r}"
    outside, incomplete = scan_source(
        source, "unit", canonical_reader=STORE_DSN_SNIPPET
    )
    assert (not outside and not incomplete) is safe, (outside, incomplete)


def test_scanner_rejects_a_reader_that_mentions_file_but_returns_startup_url() -> None:
    source = (
        "def store_dsn():\n"
        '    path = os.getenv("GPU_FAULT_STORE_URL_FILE")\n'
        "    with open(path) as handle:\n"
        "        handle.read()\n"
        '    return os.environ["GPU_FAULT_STORE_URL"]\n'
    )
    outside, incomplete = scan_source(
        source, "unit", canonical_reader=STORE_DSN_SNIPPET
    )
    assert not outside and incomplete


def test_scanner_rejects_unparseable_embedded_dsn_reads() -> None:
    source = "PROBE = 'with os.environ[\"GPU_FAULT_STORE_URL\"] as\\n'"
    outside, incomplete = scan_source(
        source, "unit", canonical_reader=STORE_DSN_SNIPPET
    )
    assert not outside and incomplete
