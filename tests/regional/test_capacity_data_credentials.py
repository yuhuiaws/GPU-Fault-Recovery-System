"""The standalone capacity probe uses the current mounted credential file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.perf import regional_capacity_data as data


class Connection:
    def __enter__(self) -> Connection:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def test_each_request_reads_current_file_instead_of_captured_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "store-url"
    first = "postgresql://placeholder@db.invalid/first"
    second = "postgresql://placeholder@db.invalid/second"
    path.write_text(first)
    path.chmod(0o600)
    monkeypatch.setenv(
        "GPU_FAULT_STORE_URL", "postgresql://placeholder@db.invalid/stale"
    )
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    observed: list[str] = []

    def connect(url: str, **kwargs: Any) -> Connection:
        observed.append(url)
        assert kwargs["connect_timeout"] == 10
        return Connection()

    monkeypatch.setattr("psycopg.connect", connect)
    monkeypatch.setattr(data, "inspect_or_cleanup", lambda *a, **k: {"total": 0})
    request = {"run_id": "run-a", "cluster_ids": ["perf-cap-000"], "cleanup": False}
    assert data.run_request(request) == {"total": 0}
    path.write_text(second)
    assert data.run_request(request) == {"total": 0}
    assert observed == [first, second], (
        "every invocation must resolve the current file value"
    )


@pytest.mark.parametrize("contents", [None, ""])
def test_configured_unreadable_or_empty_file_never_falls_back_to_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str | None
) -> None:
    path = tmp_path / "store-url"
    if contents is not None:
        path.write_text(contents)
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    monkeypatch.setenv(
        "GPU_FAULT_STORE_URL", "postgresql://placeholder@db.invalid/stale"
    )
    monkeypatch.setattr(
        "psycopg.connect",
        lambda *a, **k: pytest.fail(
            "configured FILE must not fall back to stale environment"
        ),
    )
    with pytest.raises(RuntimeError, match="credentials are unavailable"):
        data.run_request(
            {"run_id": "run-a", "cluster_ids": ["perf-cap-000"], "cleanup": False}
        )
