"""Serial, isolated PostgreSQL coverage for the legacy registry repair path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.resource_registry_scripts import DIRECT_SYNC_SCRIPT
from tests.admin.test_registry_transport import snapshot

POSTGRES_URL = os.environ.get("GPU_FAULT_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires a dedicated local PostgreSQL test URL"
)


@pytest.fixture
def database() -> Iterator[str]:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    parsed = conninfo_to_dict(POSTGRES_URL)
    assert parsed.get("host") == "127.0.0.1"
    schema = "admin_registry_" + uuid.uuid4().hex
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    url = make_conninfo(POSTGRES_URL, options=f"-c search_path={schema}")
    try:
        with psycopg.connect(url, autocommit=True) as connection:
            connection.execute(
                "CREATE TABLE gpu_fault_objects("
                "kind text NOT NULL, key text NOT NULL, payload jsonb NOT NULL, "
                "PRIMARY KEY(kind,key))"
            )
        yield url
    finally:
        with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


def run_sync(
    database: str, records: list[dict[str, Any]], *, credential_file: Path | None = None
) -> subprocess.CompletedProcess[str]:
    environment = {"GPU_FAULT_STORE_URL": database}
    if credential_file is not None:
        environment["GPU_FAULT_STORE_URL_FILE"] = str(credential_file)
    pgpass_reference = os.environ.get("PGPASSFILE")
    if pgpass_reference is not None:
        pgpass = Path(pgpass_reference)
        if not pgpass.is_absolute():
            raise ValueError("test PGPASSFILE reference must be absolute")
        if not pgpass.is_file():
            raise ValueError("test PGPASSFILE reference must name an existing file")
        environment["PGPASSFILE"] = pgpass_reference
    return subprocess.run(
        [sys.executable, "-I", "-c", DIRECT_SYNC_SCRIPT],
        input=json.dumps({"resources": records}),
        text=True,
        capture_output=True,
        env=environment,
        timeout=45,
        check=False,
    )


def rows(database: str) -> list[dict[str, Any]]:
    import psycopg

    with psycopg.connect(database) as connection:
        return [
            row[0]
            for row in connection.execute(
                "SELECT payload FROM gpu_fault_objects ORDER BY key"
            ).fetchall()
        ]


def resource() -> dict[str, Any]:
    return snapshot().resources[0].model_dump(mode="json")


@pytest.fixture
def child_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def run(arguments: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"arguments": arguments, **options})
        return subprocess.CompletedProcess(arguments, 0, "0\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


@pytest.mark.parametrize("mounted_dsn", [False, True])
def test_sync_child_forwards_only_the_explicit_absolute_pgpass_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    child_calls: list[dict[str, Any]],
    mounted_dsn: bool,
) -> None:
    pgpass = tmp_path / "test-pgpass"
    pgpass.touch(mode=0o600)
    monkeypatch.setenv("PGPASSFILE", str(pgpass))
    for name in (
        "HOME",
        "PGPASSWORD",
        "PGHOST",
        "PGHOSTADDR",
        "PGPORT",
        "PGDATABASE",
        "PGUSER",
        "PGSERVICE",
        "PGSERVICEFILE",
        "PGOPTIONS",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "KUBECONFIG",
        "GPU_FAULT_STORE_URL",
        "GPU_FAULT_STORE_URL_FILE",
    ):
        monkeypatch.setenv(name, "unit-ambient-value-must-not-be-forwarded")
    database = "host=127.0.0.1 port=1 dbname=unit_registry"
    credential_file = tmp_path / "selected-dsn" if mounted_dsn else None
    result = run_sync(database, [], credential_file=credential_file)
    expected = {"GPU_FAULT_STORE_URL": database, "PGPASSFILE": str(pgpass)}
    if credential_file is not None:
        expected["GPU_FAULT_STORE_URL_FILE"] = str(credential_file)
    assert result.returncode == 0, (
        "the fake subprocess must return its result unchanged"
    )
    assert child_calls[0]["env"] == expected, (
        "the child may receive only explicit test connection inputs and pgpass"
    )
    assert child_calls[0]["arguments"] == [
        sys.executable,
        "-I",
        "-c",
        DIRECT_SYNC_SCRIPT,
    ], "authentication must not alter the production synchronization script or argv"
    assert json.loads(child_calls[0]["input"]) == {"resources": []}, (
        "authentication references must not enter the registry payload"
    )


def test_sync_child_does_not_discover_credentials_without_an_explicit_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, child_calls: list[dict[str, Any]]
) -> None:
    monkeypatch.delenv("PGPASSFILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PGPASSWORD", "unit-ambient-value-must-not-be-forwarded")
    (tmp_path / ".pgpass").touch(mode=0o600)
    database = "host=127.0.0.1 port=1 dbname=unit_registry"
    run_sync(database, [])
    assert child_calls[0]["env"] == {"GPU_FAULT_STORE_URL": database}, (
        "missing explicit pgpass authority must not enable HOME discovery or PGPASSWORD"
    )


@pytest.mark.parametrize("reference", ["relative-pgpass", "~/.pgpass", "", "."])
def test_sync_child_rejects_relative_or_empty_pgpass_references(
    monkeypatch: pytest.MonkeyPatch, child_calls: list[dict[str, Any]], reference: str
) -> None:
    monkeypatch.setenv("PGPASSFILE", reference)
    with pytest.raises(ValueError, match="absolute"):
        run_sync("host=127.0.0.1 port=1 dbname=unit_registry", [])
    assert child_calls == [], "invalid authentication references must fail before spawn"


@pytest.mark.parametrize("directory", [False, True])
def test_sync_child_rejects_missing_or_nonfile_pgpass_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    child_calls: list[dict[str, Any]],
    directory: bool,
) -> None:
    path = tmp_path / "unavailable-pgpass"
    if directory:
        path.mkdir()
    monkeypatch.setenv("PGPASSFILE", str(path))
    with pytest.raises(ValueError, match="file"):
        run_sync("host=127.0.0.1 port=1 dbname=unit_registry", [])
    assert child_calls == [], "an unavailable pgpass file must fail before spawn"


def test_direct_sync_preserves_identity_while_updating_status(database: str) -> None:
    original = resource()
    assert run_sync(database, [original]).returncode == 0
    updated = {**original, "status": "DELETE_PENDING"}
    result = run_sync(database, [updated])
    assert result.returncode == 0, result.stderr
    assert rows(database) == [updated]


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("resource_id", "different-resource"),
        ("resource_type", "different-type"),
        ("resource_arn", "arn:aws:ec2:us-east-1:123456789012:security-group/sg-other"),
        ("provider", "different-provider"),
        ("account_id", "111122223333"),
        ("region", "us-west-2"),
        ("ownership", "EXTERNAL"),
        ("delete_policy", "PRESERVE"),
        ("dependencies", ["aws/other/resource"]),
    ],
)
def test_direct_sync_rejects_immutable_drift_atomically(
    database: str, field: str, changed: object
) -> None:
    original = resource()
    assert run_sync(database, [original]).returncode == 0
    preceding = {**original, "resource_key": "aaa/new-resource"}
    result = run_sync(database, [preceding, {**original, field: changed}])
    assert result.returncode == 1
    assert result.stderr.strip() == (
        "registry database synchronization failed: ValueError"
    )
    assert rows(database) == [original], "identity refusal committed a partial batch"


def test_concurrent_registration_cannot_replace_an_existing_identity(
    database: str,
) -> None:
    first = resource()
    second = {**first, "resource_id": "competing-resource"}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_sync, database, [value]) for value in (first, second)
        ]
        results = [future.result(timeout=45) for future in futures]
    assert sorted(result.returncode for result in results) == [0, 1]
    assert rows(database) in ([first], [second])


def test_direct_sync_prefers_current_mounted_credentials(
    database: str, tmp_path: Path
) -> None:
    credential = tmp_path / "dsn"
    credential.write_text(database, encoding="utf-8")
    credential.chmod(0o600)
    original = resource()
    result = run_sync("unusable-startup-dsn", [original], credential_file=credential)
    assert result.returncode == 0, result.stderr
    assert rows(database) == [original]


@pytest.mark.parametrize("exists", [False, True])
def test_unreadable_or_empty_credential_file_does_not_use_old_env_credentials(
    database: str, tmp_path: Path, exists: bool
) -> None:
    credential = tmp_path / "dsn"
    if exists:
        credential.write_text("", encoding="utf-8")
    result = run_sync(database, [resource()], credential_file=credential)
    assert result.returncode == 1
    assert rows(database) == []


def test_database_errors_do_not_expose_row_or_connection_details(database: str) -> None:
    import psycopg

    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute(
            """CREATE FUNCTION reject_sync() RETURNS trigger LANGUAGE plpgsql
               AS $$ BEGIN RAISE EXCEPTION 'private-database-sentinel'; END $$"""
        )
        connection.execute(
            "CREATE TRIGGER reject_sync BEFORE INSERT ON gpu_fault_objects "
            "FOR EACH ROW EXECUTE FUNCTION reject_sync()"
        )
    result = run_sync(database, [resource()])
    assert result.returncode == 1
    assert result.stderr.strip() == (
        "registry database synchronization failed: RaiseException"
    ), "the trigger must execute; an authentication failure cannot satisfy this test"
    assert "private-database-sentinel" not in result.stdout + result.stderr
    assert database not in result.stdout + result.stderr


def test_reinstall_generations_preserve_old_rows_and_identity_guards(
    database: str,
) -> None:
    from gpu_fault.installation_lifecycle import generation_registry_site_id

    original = resource()
    first_scope = generation_registry_site_id(original["site_id"], "a" * 32)
    second_scope = generation_registry_site_id(original["site_id"], "b" * 32)
    first = {
        **original,
        "site_id": first_scope,
        "resource_id": "recreated-first-resource",
    }
    second = {
        **original,
        "site_id": second_scope,
        "resource_id": "recreated-second-resource",
    }
    for value in (original, first, second):
        assert run_sync(database, [value]).returncode == 0
    observed = {row["site_id"]: row for row in rows(database)}
    assert observed == {
        original["site_id"]: original,
        first_scope: first,
        second_scope: second,
    }
    conflict = {**second, "resource_id": "different-resource-in-the-same-generation"}
    assert run_sync(database, [conflict]).returncode == 1
    assert {row["site_id"]: row for row in rows(database)} == observed
