from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault import store_migrate
from scripts.e2e.regional.boot_store_lifecycle_evidence import STORE_PROBE

FILE_DSN = "postgresql://projected.invalid/control"
ENV_DSN = "postgresql://startup.invalid/control"


class ConnectionReached(Exception):
    pass


@pytest.fixture(params=["cli", "probe"])
def credential_consumer(request, monkeypatch):
    if request.param == "cli":
        monkeypatch.setattr(sys, "argv", ["store-migrate", "--schema-preflight"])
        return store_migrate.main
    return lambda: exec(STORE_PROBE, {"INPUT": '{"keys": []}'})


@pytest.fixture
def connections(monkeypatch):
    seen = []

    def connect(dsn, **kwargs):
        seen.append((dsn, kwargs))
        raise ConnectionReached

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect))
    return seen


@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_one_successful_projected_read_is_used_without_a_cached_reread(
    tmp_path, monkeypatch, credential_consumer, connections, damage
):
    path = tmp_path / "dsn"
    path.write_text(FILE_DSN + "\n", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", ENV_DSN)
    original_read = Path.read_text
    reads = []

    def read_then_remove(self, *args, **kwargs):
        if self != path:
            return original_read(self, *args, **kwargs)
        reads.append(self)
        value = original_read(self, *args, **kwargs)
        if damage == "missing":
            self.unlink()
        else:
            self.write_text("", encoding="utf-8")
        return value

    monkeypatch.setattr(Path, "read_text", read_then_remove)
    with pytest.raises(ConnectionReached):
        credential_consumer()
    assert reads == [path]
    assert [dsn for dsn, _kwargs in connections] == [FILE_DSN]

    with pytest.raises((SystemExit, RuntimeError)) as raised:
        credential_consumer()
    if isinstance(raised.value, SystemExit):
        assert raised.value.code == 2
    assert reads == [path, path]
    assert len(connections) == 1


@pytest.mark.parametrize(
    "damage", ["missing", "empty", "whitespace", "invalid-utf8", "unreadable"]
)
def test_bad_projection_refuses_before_database_io_without_env_fallback(
    tmp_path, monkeypatch, capsys, credential_consumer, connections, damage
):
    path = tmp_path / "dsn"
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", ENV_DSN)
    if damage == "invalid-utf8":
        path.write_bytes(b"\xff")
    elif damage != "missing":
        path.write_text(" \n\t" if damage == "whitespace" else "", encoding="utf-8")
    if damage == "unreadable":
        original_read = Path.read_text

        def denied(self, *args, **kwargs):
            if self == path:
                raise PermissionError("private-read-error-detail")
            return original_read(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises((SystemExit, RuntimeError)) as raised:
        credential_consumer()
    if isinstance(raised.value, SystemExit):
        assert raised.value.code == 2
    assert not connections, (
        "invalid projected credentials must fail before any database connection"
    )
    output = capsys.readouterr()
    for forbidden in (FILE_DSN, ENV_DSN, "private-read-error-detail"):
        assert forbidden not in output.out + output.err + str(raised.value)


def test_unconfigured_projection_allows_the_explicit_legacy_environment(
    monkeypatch, credential_consumer, connections
):
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", ENV_DSN)
    with pytest.raises(ConnectionReached):
        credential_consumer()
    assert [dsn for dsn, _kwargs in connections] == [ENV_DSN]


def test_each_new_invocation_reads_the_rotated_projection(
    tmp_path, monkeypatch, credential_consumer, connections
):
    path = tmp_path / "dsn"
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", ENV_DSN)
    values = [FILE_DSN, "postgresql://rotated.invalid/control"]
    for value in values:
        path.write_text(value + "\n", encoding="utf-8")
        with pytest.raises(ConnectionReached):
            credential_consumer()
    assert [dsn for dsn, _kwargs in connections] == values
