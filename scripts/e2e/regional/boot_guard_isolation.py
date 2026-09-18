"""Database and Pod isolation for the BOOT startup-guard fixture.

The database entrypoint is sent to a CPU Pod on private stdin. It never
initializes a Store until the server has confirmed the disposable database.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlsplit, urlunsplit

PRODUCTION_DATABASE = "gpu_fault"
PROBE_DATABASE = "gpu_fault_guardprobe"
PROBE_SECRET = "gpu-fault-aurora-guardprobe"
PROBE_VOLUME = "guardprobe-store"
PROBE_MOUNT = "/etc/gpu-fault/guardprobe"
STORE_URL = "GPU_FAULT_STORE_URL"
STORE_URL_FILE = "GPU_FAULT_STORE_URL_FILE"


class BootGuardError(RuntimeError):
    pass


def database_urls() -> tuple[str, str]:
    path = os.environ.get(STORE_URL_FILE, "").strip()
    value = (
        Path(path).read_text(encoding="utf-8").strip()
        if path
        else os.environ.get(STORE_URL, "").strip()
    )
    try:
        parts = urlsplit(value)
        valid = (
            parts.scheme in {"postgresql", "postgres"}
            and bool(parts.hostname)
            and parts.path == f"/{PRODUCTION_DATABASE}"
            and not parts.fragment
            and not {
                "dbname",
                "host",
                "hostaddr",
                "port",
                "service",
                "servicefile",
            }.intersection(key for key, _ in parse_qsl(parts.query))
        )
    except ValueError:
        valid = False
    if not valid:
        raise BootGuardError(
            "BOOT probe requires an explicit production database reference"
        )
    return (
        urlunsplit(parts._replace(path="/postgres")),
        urlunsplit(parts._replace(path=f"/{PROBE_DATABASE}")),
    )


def require_database(connection: Any, expected: str) -> None:
    if connection.execute("SELECT current_database()").fetchone() != (expected,):
        raise BootGuardError("database identity differs from the BOOT probe target")


@contextmanager
def isolated_store_environment(url: str) -> Iterator[None]:
    names = {STORE_URL, STORE_URL_FILE} | {
        name for name in os.environ if name.startswith("PG")
    }
    saved = {name: os.environ[name] for name in names if name in os.environ}
    try:
        for name in names:
            os.environ.pop(name, None)
        os.environ[STORE_URL] = url
        yield
    finally:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update(saved)


def initialize_database(*, reset: bool = False) -> None:
    import psycopg
    from psycopg import sql

    from gpu_fault.store import PostgresStore

    admin, probe = database_urls()
    with isolated_store_environment(probe):
        with psycopg.connect(admin, autocommit=True, connect_timeout=10) as connection:
            require_database(connection, "postgres")
            exists = connection.execute(
                "SELECT 1 FROM pg_database WHERE datname=%s", (PROBE_DATABASE,)
            ).fetchone()
            if not exists:
                connection.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(PROBE_DATABASE))
                )
        with psycopg.connect(probe, autocommit=True, connect_timeout=10) as connection:
            require_database(connection, PROBE_DATABASE)
            if reset:
                # A durable registry head must not survive into the next variant.
                connection.execute("DROP SCHEMA public CASCADE")
                connection.execute("CREATE SCHEMA public")
        store = PostgresStore(
            probe,
            pool_min_size=0,
            pool_max_size=1,
            pool_timeout_seconds=10,
            initialize_schema=True,
        )
        store.close()
        with psycopg.connect(probe, autocommit=True, connect_timeout=10) as connection:
            require_database(connection, PROBE_DATABASE)
            row = connection.execute(
                "SELECT to_regclass('gpu_fault_schema_version')"
            ).fetchone()
            if row is None or row[0] is None:
                raise BootGuardError(
                    "BOOT probe schema initialization was not verified"
                )


def drop_database() -> None:
    import psycopg
    from psycopg import sql

    admin, probe = database_urls()
    with isolated_store_environment(probe):
        with psycopg.connect(admin, autocommit=True, connect_timeout=10) as connection:
            require_database(connection, "postgres")
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(
                    sql.Identifier(PROBE_DATABASE)
                )
            )
            if (
                connection.execute(
                    "SELECT 1 FROM pg_database WHERE datname=%s", (PROBE_DATABASE,)
                ).fetchone()
                is not None
            ):
                raise BootGuardError("BOOT probe database remains after cleanup")


def isolate_probe_store(pod_spec: dict[str, Any]) -> None:
    containers = pod_spec.get("containers")
    if (
        not isinstance(containers, list)
        or len(containers) != 1
        or pod_spec.get("initContainers")
        or pod_spec.get("ephemeralContainers")
    ):
        raise BootGuardError("BOOT probe requires exactly one application container")
    container = containers[0]
    entries = container.get("env", [])
    environment = {entry["name"]: entry for entry in entries}
    if len(environment) != len(entries):
        raise BootGuardError("BOOT probe environment repeats a variable")
    reference = (
        environment.get(STORE_URL, {}).get("valueFrom", {}).get("secretKeyRef", {})
    )
    if not reference.get("name") or reference.get("key") != "postgres-url":
        raise BootGuardError("production Store must use the declared DSN Secret key")
    production_secrets = {reference["name"], "gpu-fault-aurora"}
    for source in container.get("envFrom", []):
        if source.get("secretRef", {}).get("name") in production_secrets:
            raise BootGuardError(
                "production DSN Secret cannot be imported through envFrom"
            )
    for name, entry in environment.items():
        if (
            name != STORE_URL
            and entry.get("valueFrom", {}).get("secretKeyRef", {}).get("name")
            in production_secrets
        ):
            raise BootGuardError(
                "production DSN Secret is referenced by another variable"
            )
    volumes = pod_spec.get("volumes", [])
    original_file = environment.get(STORE_URL_FILE, {}).get("value", "")
    file_volumes = {
        mount["name"]
        for mount in container.get("volumeMounts", [])
        if original_file
        and (
            original_file == mount["mountPath"]
            or original_file.startswith(mount["mountPath"].rstrip("/") + "/")
        )
    }
    removed = {
        volume["name"]
        for volume in volumes
        if volume["name"] in file_volumes
        or volume.get("secret", {}).get("secretName") in production_secrets
        or any(
            source.get("secret", {}).get("name") in production_secrets
            for source in volume.get("projected", {}).get("sources", [])
        )
    }
    remaining = [volume for volume in volumes if volume["name"] not in removed]
    mounts = [
        mount
        for mount in container.get("volumeMounts", [])
        if mount["name"] not in removed
    ]
    if any(volume["name"] == PROBE_VOLUME for volume in remaining) or any(
        mount["mountPath"] == PROBE_MOUNT for mount in mounts
    ):
        raise BootGuardError("BOOT probe Store mount collides with the application")
    pod_spec["volumes"] = [
        *remaining,
        {
            "name": PROBE_VOLUME,
            "secret": {"secretName": PROBE_SECRET, "defaultMode": 0o444},
        },
    ]
    container["volumeMounts"] = [
        *mounts,
        {"name": PROBE_VOLUME, "mountPath": PROBE_MOUNT, "readOnly": True},
    ]
    environment[STORE_URL] = {
        "name": STORE_URL,
        "valueFrom": {"secretKeyRef": {"name": PROBE_SECRET, "key": "postgres-url"}},
    }
    environment[STORE_URL_FILE] = {
        "name": STORE_URL_FILE,
        "value": f"{PROBE_MOUNT}/postgres-url",
    }
    container["env"] = list(environment.values())


def main() -> int:
    try:
        operation = sys.argv[1]
        if operation in {"initialize", "reset"}:
            initialize_database(reset=operation == "reset")
            print("guardprobe schema initialized")
        elif operation == "drop":
            drop_database()
            print("guardprobe database absent")
        elif operation == "secret":
            _, probe = database_urls()
            # This branch is piped directly into kubectl apply, never evidence.
            print(
                json.dumps(
                    {
                        "apiVersion": "v1",
                        "kind": "Secret",
                        "metadata": {"name": PROBE_SECRET},
                        "type": "Opaque",
                        "data": {
                            "postgres-url": base64.b64encode(probe.encode()).decode()
                        },
                    }
                )
            )
        else:
            raise BootGuardError("unknown BOOT database operation")
    except Exception as exc:
        print(f"BOOT database operation failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
