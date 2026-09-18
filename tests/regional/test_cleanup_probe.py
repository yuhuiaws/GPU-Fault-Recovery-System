from __future__ import annotations

from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module

PROBE = lazy_script_module(
    Path(__file__).resolve().parents[2]
    / "deploy/control-plane/tools/cleanup_activity.py"
)


def test_cleanup_probe_uses_current_mounted_database_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mounted = tmp_path / "postgres-url"
    mounted.write_text(
        "host=current.invalid user=test dbname=test sslmode=verify-full sslrootcert=/private/ca"
    )
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(mounted))
    monkeypatch.setenv(
        "GPU_FAULT_STORE_URL",
        "host=stale.invalid sslmode=verify-full sslrootcert=/private/ca",
    )
    arguments = PROBE.connection_arguments()
    assert arguments["host"] == "current.invalid"
    assert arguments["connect_timeout"] == 10
    assert "default_transaction_read_only=on" in arguments["options"]
    assert "statement_timeout=" in arguments["options"]
    assert "lock_timeout=" in arguments["options"]


def test_unreadable_mounted_database_configuration_does_not_use_stale_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "host=stale.invalid")
    with pytest.raises(OSError):
        PROBE.connection_arguments()


@pytest.mark.parametrize("mode", ["disable", "require", "verify-ca"])
def test_cleanup_probe_never_weakens_database_tls(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", f"host=db.invalid sslmode={mode}")
    with pytest.raises(ValueError, match="verified database TLS"):
        PROBE.connection_arguments()
