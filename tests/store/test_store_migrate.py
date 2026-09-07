from __future__ import annotations

import json
import os
import sys

import pytest

from gpu_fault import store_migrate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")


def test_store_migrate_can_initialize_schema(monkeypatch, capsys) -> None:
    calls = []

    class FakeStore:
        def __init__(self, url, **kwargs) -> None:
            calls.append(("open", url, kwargs))

        def close(self) -> None:
            calls.append(("close", None))

    monkeypatch.setattr(store_migrate, "PostgresStore", FakeStore)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--ensure-schema",
            "--postgres-url",
            "postgresql://db/gpu_fault",
        ],
    )

    store_migrate.main()

    assert calls == [
        ("open", "postgresql://db/gpu_fault", {"hot_state_mode": "legacy"}),
        ("close", None),
    ]
    assert "schema initialization complete" in capsys.readouterr().out


def test_store_migrate_can_backfill_hot_state(monkeypatch, capsys) -> None:
    calls = []

    class FakeStore:
        def __init__(self, url, **kwargs) -> None:
            calls.append(("open", url, kwargs))

        def backfill_hot_state_tables(self):
            return {"gpu_metric_latest": 3}

        def hot_state_migration_status(self):
            return {
                "gpu_metric_latest": {
                    "legacy": 3,
                    "dedicated": 3,
                    "matched_keys": 3,
                    "missing_or_mismatched": 0,
                }
            }

        def close(self) -> None:
            calls.append(("close", None))

    monkeypatch.setattr(store_migrate, "PostgresStore", FakeStore)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--backfill-hot-state",
            "--postgres-url",
            "postgresql://db/gpu_fault",
        ],
    )

    store_migrate.main()

    body = json.loads(capsys.readouterr().out)
    assert body["backfill"]["gpu_metric_latest"] == 3
    assert body["status"]["gpu_metric_latest"]["missing_or_mismatched"] == 0
    assert calls == [
        ("open", "postgresql://db/gpu_fault", {"hot_state_mode": "dual"}),
        ("close", None),
    ]


def test_store_migrate_can_finalize_counter_shards(monkeypatch, capsys) -> None:
    calls = []

    class FakeStore:
        def __init__(self, url, **kwargs) -> None:
            calls.append(("open", url, kwargs))

        def finalize_processor_counter_shards(self):
            return {"mode": "partitioned", "ready": True}

        def close(self) -> None:
            calls.append(("close", None))

    monkeypatch.setattr(store_migrate, "PostgresStore", FakeStore)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--finalize-processor-counter-shards",
            "--postgres-url",
            "postgresql://db/gpu_fault",
        ],
    )

    store_migrate.main()

    assert json.loads(capsys.readouterr().out) == {"mode": "partitioned", "ready": True}
    assert calls == [("open", "postgresql://db/gpu_fault", {}), ("close", None)]


def test_store_migrate_can_restore_legacy_counters(monkeypatch, capsys) -> None:
    class FakeStore:
        def __init__(self, _url, **_kwargs) -> None:
            pass

        def restore_legacy_processor_counters(self):
            return {"mode": "dual", "ready": True}

        def close(self) -> None:
            pass

    monkeypatch.setattr(store_migrate, "PostgresStore", FakeStore)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--restore-legacy-processor-counters",
            "--postgres-url",
            "postgresql://db/gpu_fault",
        ],
    )

    store_migrate.main()

    assert json.loads(capsys.readouterr().out)["mode"] == "dual"


def test_store_migrate_refuses_unsafe_legacy_purge(monkeypatch) -> None:
    class FakeStore:
        def __init__(self, _url, **_kwargs) -> None:
            pass

        def purge_legacy_hot_state(self):
            raise RuntimeError("hot state backfill is incomplete")

        def close(self) -> None:
            pass

    monkeypatch.setattr(store_migrate, "PostgresStore", FakeStore)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--purge-legacy-hot-state",
            "--postgres-url",
            "postgresql://db/gpu_fault",
        ],
    )

    try:
        store_migrate.main()
    except RuntimeError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("unsafe hot-state purge was accepted")


@pytest.fixture
def without_pg_stat_statements():
    """Leave the extension state as the other tests found it."""

    psycopg = pytest.importorskip("psycopg")
    assert POSTGRES_URL is not None

    def drop() -> None:
        with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("DROP EXTENSION IF EXISTS pg_stat_statements")

    drop()
    yield
    drop()


@pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
def test_store_migrate_ensure_diagnostics_installs_pg_stat_statements(
    monkeypatch, capsys, without_pg_stat_statements
) -> None:
    """Store review 2026-09-07, item K2. The postgres:16 image ships contrib
    and the test role is a superuser, so the extension installs; the report
    then shows it and drops the matching warning."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--ensure-diagnostics",
            "--postgres-url",
            POSTGRES_URL,
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        store_migrate.main()

    assert exit_info.value.code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["diagnostics"]["pg_stat_statements_installed"] is True
    assert all("pg_stat_statements" not in item for item in body["warnings"]), body[
        "warnings"
    ]
    assert set(body["diagnostics"]) >= {
        "log_lock_waits",
        "deadlock_timeout",
        "log_min_duration_statement",
        "shared_preload_libraries",
    }


def test_store_migrate_ensure_diagnostics_reports_insufficient_privilege(
    monkeypatch, capsys
) -> None:
    psycopg = pytest.importorskip("psycopg")

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            return None

        def execute(self, query, _params=None):
            if query.startswith("CREATE EXTENSION"):
                raise psycopg.errors.InsufficientPrivilege(
                    "permission denied to create extension"
                )
            raise AssertionError(f"unexpected statement after the failure: {query}")

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            return None

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(psycopg, "connect", lambda *_args, **_kwargs: FakeConnection())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--ensure-diagnostics",
            "--postgres-url",
            "postgresql://db/gpu_fault",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        store_migrate.main()

    assert exit_info.value.code == 1
    assert "insufficient privilege" in capsys.readouterr().out
