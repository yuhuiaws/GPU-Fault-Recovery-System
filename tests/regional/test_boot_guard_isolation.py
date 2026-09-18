from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from gpu_fault import store
from scripts.e2e.regional import boot_guard_isolation as guard


def pod_spec() -> dict[str, Any]:
    return {
        "containers": [
            {
                "name": "api",
                "env": [
                    {
                        "name": guard.STORE_URL,
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "gpu-fault-aurora",
                                "key": "postgres-url",
                            }
                        },
                    },
                    {"name": guard.STORE_URL_FILE, "value": "/production/postgres-url"},
                ],
                "volumeMounts": [
                    {"name": "aurora", "mountPath": "/production", "readOnly": True},
                    {"name": "ca", "mountPath": "/ca", "readOnly": True},
                ],
            }
        ],
        "volumes": [
            {"name": "aurora", "secret": {"secretName": "gpu-fault-aurora"}},
            {"name": "ca", "configMap": {"name": "rds-public-ca"}},
        ],
    }


@pytest.mark.parametrize("projected", [False, True])
def test_probe_cannot_mount_the_production_dsn(projected: bool) -> None:
    spec = pod_spec()
    if projected:
        spec["volumes"][0] = {
            "name": "aurora",
            "projected": {"sources": [{"secret": {"name": "gpu-fault-aurora"}}]},
        }

    guard.isolate_probe_store(spec)

    container = spec["containers"][0]
    environment = {entry["name"]: entry for entry in container["env"]}
    assert environment[guard.STORE_URL]["valueFrom"]["secretKeyRef"]["name"] == (
        guard.PROBE_SECRET
    ), "the startup URL must reference only the disposable database"
    assert environment[guard.STORE_URL_FILE]["value"] == (
        f"{guard.PROBE_MOUNT}/postgres-url"
    ), "credential refresh must not override the isolated URL"
    assert spec["volumes"] == [
        {"name": "ca", "configMap": {"name": "rds-public-ca"}},
        {
            "name": guard.PROBE_VOLUME,
            "secret": {"secretName": guard.PROBE_SECRET, "defaultMode": 0o444},
        },
    ], "no production DSN volume may reach the probe"
    assert container["volumeMounts"] == [
        {"name": "ca", "mountPath": "/ca", "readOnly": True},
        {"name": guard.PROBE_VOLUME, "mountPath": guard.PROBE_MOUNT, "readOnly": True},
    ], "the public TLS trust bundle must survive isolation"


@pytest.mark.parametrize("route", ["envFrom", "alias", "sidecar", "init"])
def test_unmodeled_production_secret_routes_fail_closed(route: str) -> None:
    spec = pod_spec()
    container = spec["containers"][0]
    if route == "envFrom":
        container["envFrom"] = [{"secretRef": {"name": "gpu-fault-aurora"}}]
    elif route == "alias":
        container["env"].append(
            {
                "name": "ANOTHER_DATABASE",
                "valueFrom": {
                    "secretKeyRef": {"name": "gpu-fault-aurora", "key": "postgres-url"}
                },
            }
        )
    elif route == "sidecar":
        spec["containers"].append({"name": "sidecar"})
    else:
        spec["initContainers"] = [{"name": "init"}]

    with pytest.raises(guard.BootGuardError):
        guard.isolate_probe_store(spec)


class Database:
    def __init__(
        self, name: str, calls: list[str], *, schema_present: bool = True
    ) -> None:
        self.name = name
        self.calls = calls
        self.schema_present = schema_present
        self.row: tuple[Any, ...] | None = None

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_args: Any) -> None:
        pass

    def execute(self, statement: Any, *_args: Any) -> Database:
        if not isinstance(statement, str):
            statement = statement.as_string()
        self.calls.append(statement)
        self.row = (self.name,) if statement == "SELECT current_database()" else None
        if statement == "SELECT to_regclass('gpu_fault_schema_version')":
            self.row = ("gpu_fault_schema_version" if self.schema_present else None,)
        return self

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.row


@pytest.mark.parametrize("reset", [False, True])
def test_schema_initialization_cannot_follow_the_inherited_dsn_file(
    reset: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = tmp_path / "projected-dsn"
    credential.write_text("postgresql://fresh.invalid/gpu_fault", encoding="utf-8")
    monkeypatch.setenv(guard.STORE_URL, "postgresql://stale.invalid/gpu_fault")
    monkeypatch.setenv(guard.STORE_URL_FILE, str(credential))
    monkeypatch.setenv("PGSERVICE", "another-database")
    calls: list[str] = []

    def connect(url: str, **_kwargs: Any) -> Database:
        assert urlsplit(url).hostname == "fresh.invalid", "rotated credentials must win"
        assert guard.STORE_URL_FILE not in guard.os.environ, (
            "the pool must not see the production DSN file"
        )
        assert "PGSERVICE" not in guard.os.environ, "libpq may not inherit routing"
        return Database(urlsplit(url).path.lstrip("/"), calls)

    def initialize(url: str, **_kwargs: Any) -> Any:
        assert calls[-1] == (
            "CREATE SCHEMA public" if reset else "SELECT current_database()"
        ), "actual database identity must be checked before schema initialization"
        assert urlsplit(url).path == f"/{guard.PROBE_DATABASE}", (
            "Store must stay isolated"
        )
        assert guard.STORE_URL_FILE not in guard.os.environ, (
            "file precedence must stay disabled"
        )
        calls.append("store")
        return type("Store", (), {"close": lambda self: calls.append("close")})()

    monkeypatch.setattr("psycopg.connect", connect)
    monkeypatch.setattr(store, "PostgresStore", initialize)

    guard.initialize_database(reset=reset)

    assert calls[-4:] == [
        "store",
        "close",
        "SELECT current_database()",
        "SELECT to_regclass('gpu_fault_schema_version')",
    ], "the isolated pool must be closed before independently verifying its schema"
    assert guard.os.environ[guard.STORE_URL_FILE] == str(credential), (
        "the enclosing CPU probe environment must be restored"
    )


def test_successful_initializer_without_probe_schema_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(guard.STORE_URL_FILE, raising=False)
    monkeypatch.setenv(guard.STORE_URL, "postgresql://example.invalid/gpu_fault")
    calls: list[str] = []
    monkeypatch.setattr(
        "psycopg.connect",
        lambda url, **_kwargs: Database(
            urlsplit(url).path.lstrip("/"), calls, schema_present=False
        ),
    )
    monkeypatch.setattr(
        store,
        "PostgresStore",
        lambda *_args, **_kwargs: type(
            "Store", (), {"close": lambda self: calls.append("close")}
        )(),
    )

    with pytest.raises(guard.BootGuardError, match="initialization was not verified"):
        guard.initialize_database()

    assert "close" in calls
    assert calls[-1] == "SELECT to_regclass('gpu_fault_schema_version')"


@pytest.mark.parametrize("wrong_connection", ["admin", "probe"])
def test_wrong_actual_database_cannot_run_ddl(
    wrong_connection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(guard.STORE_URL_FILE, raising=False)
    monkeypatch.setenv(guard.STORE_URL, "postgresql://example.invalid/gpu_fault")
    calls: list[str] = []

    def connect(url: str, **_kwargs: Any) -> Database:
        expected = urlsplit(url).path.lstrip("/")
        wrong = expected == (
            "postgres" if wrong_connection == "admin" else guard.PROBE_DATABASE
        )
        return Database("gpu_fault" if wrong else expected, calls)

    monkeypatch.setattr("psycopg.connect", connect)
    monkeypatch.setattr(
        store, "PostgresStore", lambda *_a, **_k: pytest.fail("Store must not open")
    )

    with pytest.raises(guard.BootGuardError, match="identity"):
        guard.initialize_database(reset=True)

    assert "DROP SCHEMA public CASCADE" not in calls, (
        "production schema must remain untouched"
    )
    if wrong_connection == "admin":
        assert 'CREATE DATABASE "gpu_fault_guardprobe"' not in calls, (
            "unknown admin identity must refuse database creation too"
        )


@pytest.mark.parametrize(
    "value", ["", "postgresql://example.invalid/gpu_fault?dbname=other"]
)
def test_empty_or_redirected_reference_never_connects(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(guard.STORE_URL_FILE, raising=False)
    monkeypatch.setenv(guard.STORE_URL, value)
    monkeypatch.setattr(
        "psycopg.connect", lambda *_a, **_k: pytest.fail("must not connect")
    )

    with pytest.raises(guard.BootGuardError, match="reference"):
        guard.initialize_database()
