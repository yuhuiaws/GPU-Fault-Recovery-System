from __future__ import annotations

import json
import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from gpu_fault.store import PostgresStore
from scripts.e2e.regional.run_cap005_postgres_suite import database_url, validate_server

ROOT = Path(__file__).resolve().parents[2]


def validated_grant() -> str:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("NOTIFY008 PostgreSQL process tests require explicit serial -n0")
    directory = ROOT / ".codex"
    override = os.environ.get("PYTEST_GPU_FAULT_POSTGRES_ALLOCATION_DIR")
    if override is not None:
        if not override.strip():
            pytest.fail("NOTIFY008 PostgreSQL allocation directory override is empty")
        directory = Path(override)
    try:
        owner = json.loads((directory / "cov95-postgres-owner.json").read_text())
        url = (directory / "cov95-postgres-url").read_text().strip()
    except (OSError, ValueError):
        pytest.fail("NOTIFY008 PostgreSQL allocation files are missing or invalid")
    if not isinstance(owner, dict):
        pytest.fail("NOTIFY008 PostgreSQL grant metadata is not an object")
    port_text = str(owner.get("port", ""))
    if not re.fullmatch(r"[1-9][0-9]{0,4}", port_text) or int(port_text) > 65535:
        pytest.fail("NOTIFY008 PostgreSQL grant port is invalid")
    port = int(port_text)
    parsed = urlsplit(url)
    if (
        os.environ.get("GPU_FAULT_TEST_POSTGRES_URL") != url
        or parsed.hostname != "127.0.0.1"
        or parsed.port != port
        or parsed.password is not None
        or not isinstance(owner.get("owner"), str)
        or not owner["owner"].strip()
        or re.fullmatch(r"[0-9a-f]{64}", owner.get("container", "")) is None
        or os.environ.get("HOME") != owner.get("pgpass_home")
        or os.environ.get("PGPASSFILE") != str(Path(owner["pgpass_home"]) / ".pgpass")
    ):
        pytest.fail("NOTIFY008 PostgreSQL grant identity differs")
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            '{"id":{{json .Id}},"running":{{json .State.Running}},'
            '"labels":{{json .Config.Labels}},"ports":{{json .NetworkSettings.Ports}}}',
            owner["container"],
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    actual = json.loads(result.stdout)
    if (
        actual.get("id") != owner["container"]
        or actual.get("running") is not True
        or actual.get("labels", {}).get("gpu-fault.test-owner") != owner["owner"]
        or actual.get("ports", {}).get("5432/tcp")
        != [{"HostIp": "127.0.0.1", "HostPort": str(port)}]
    ):
        pytest.fail("NOTIFY008 PostgreSQL container ownership or port binding differs")
    validate_server(url)
    return url


@dataclass(frozen=True)
class GrantedFactory:
    url: str

    def __call__(self):
        return PostgresStore(
            self.url,
            initialize_schema=False,
            pool_min_size=1,
            pool_max_size=2,
            pool_timeout_seconds=2,
        )


@contextmanager
def isolated_database():
    import psycopg
    from psycopg import sql

    base_url = validated_grant()
    name = f"notify008_{uuid4().hex[:16]}"
    oid = None
    creation_started = False
    try:
        with psycopg.connect(base_url, autocommit=True) as connection:
            assert (
                connection.execute(
                    "SELECT oid FROM pg_database WHERE datname=%s", (name,)
                ).fetchone()
                is None
            ), "the test must not adopt an existing database"
            creation_started = True
            connection.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name))
            )
            (oid,) = connection.execute(
                "SELECT oid FROM pg_database WHERE datname=%s", (name,)
            ).fetchone()
        url = database_url(base_url, name)
        setup = PostgresStore(
            url, initialize_schema=True, pool_min_size=1, pool_max_size=2
        )
        setup.close()
        yield GrantedFactory(url)
    finally:
        assert validated_grant() == base_url, (
            "the PostgreSQL grant changed before cleanup"
        )
        with psycopg.connect(base_url, autocommit=True) as connection:
            current = connection.execute(
                "SELECT oid FROM pg_database WHERE datname=%s", (name,)
            ).fetchone()
            if current is not None and creation_started:
                assert oid is not None, (
                    "missing creation identity requires owner review, "
                    "not a blind database drop"
                )
                assert current == (oid,), "cleanup must not drop a replaced database"
                connection.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                        sql.Identifier(name)
                    )
                )
            if creation_started:
                assert (
                    connection.execute(
                        "SELECT oid FROM pg_database WHERE datname=%s", (name,)
                    ).fetchone()
                    is None
                ), "the private NOTIFY008 test database must be absent"
