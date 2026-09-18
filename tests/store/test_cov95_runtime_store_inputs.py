from __future__ import annotations

import builtins
from datetime import datetime

import pytest
from pydantic import BaseModel

from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.pool import StoreCredentials
from gpu_fault.store.postgres.state_table_payload import (
    STATE_LAYOUTS,
    join_state_record,
    split_state_record,
    state_update_columns,
)
from tests._builders import workflow_request


@pytest.fixture
def no_pool(monkeypatch):
    import psycopg_pool

    calls = []

    def blocked_pool(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("invalid Store configuration attempted to open a connection pool")

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", blocked_pool)
    return calls


@pytest.mark.parametrize(
    "options",
    [
        {"pool_min_size": -1},
        {"pool_max_size": 0},
        {"pool_min_size": 2, "pool_max_size": 1},
        {"pool_timeout_seconds": 0},
    ],
)
def test_invalid_pool_limits_fail_before_opening_transport(no_pool, options):
    with pytest.raises(ValueError, match="invalid PostgreSQL pool"):
        PostgresStore("postgresql://fake.invalid/postgres", **options)
    assert no_pool == [], "invalid pool bounds must not reach the transport factory"


@pytest.mark.parametrize("setting", ["hot", "queue"])
def test_unknown_storage_mode_cannot_open_a_pool(no_pool, monkeypatch, setting):
    options = {"hot_state_mode": "legacy"}
    if setting == "hot":
        options["hot_state_mode"] = "unknown"
    else:
        monkeypatch.setenv("GPU_FAULT_PROCESSOR_QUEUE_STATE_MODE", "unknown")
    with pytest.raises(ValueError, match="must be"):
        PostgresStore("postgresql://fake.invalid/postgres", **options)
    assert no_pool == [], "unknown mode must be refused before database access"


def test_missing_optional_pool_dependency_has_an_actionable_error(monkeypatch):
    real_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "psycopg_pool":
            raise ImportError("synthetic optional dependency absence")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", unavailable)
        with pytest.raises(
            RuntimeError, match=r"install gpu-fault-control-plane\[postgres\]"
        ):
            PostgresStore("postgresql://fake.invalid/postgres")


@pytest.mark.parametrize("value", ["", "not-a-dsn", "postgresql://["])
def test_dsn_diagnostics_fail_closed_for_unparseable_input(value):
    assert StoreCredentials(value, path=None).redacted() == "<unparsable dsn>"


def test_dsn_diagnostics_do_not_require_a_port_or_authentication_material():
    credentials = StoreCredentials("postgresql://unit@localhost/postgres", path=None)
    assert credentials.redacted() == "postgresql://unit:***@localhost/postgres"


class IncompleteWorkflow(BaseModel):
    request_id: str


def test_state_split_requires_every_declared_mutable_field():
    with pytest.raises(ValueError, match="state fields are missing"):
        split_state_record(
            STATE_LAYOUTS["workflow"], IncompleteWorkflow(request_id="incomplete")
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"created_at_naive": "false"},
        {"created_at": "not-a-timestamp"},
        {"created_at": datetime(2026, 9, 12)},
        {"created_at": None, "created_at_naive": True},
    ],
)
def test_state_join_rejects_ambiguous_timestamp_columns(changed):
    layout = STATE_LAYOUTS["workflow"]
    columns = split_state_record(layout, workflow_request("example", "incident"))
    columns.update(changed)
    with pytest.raises(ValueError, match="timestamp"):
        join_state_record(layout, columns)


@pytest.mark.parametrize(
    "fields", [frozenset(), frozenset({"request_id"}), frozenset({"unknown"})]
)
def test_partial_state_updates_cannot_replace_identity_or_unknown_fields(fields):
    with pytest.raises(ValueError, match="invalid columns"):
        state_update_columns(
            STATE_LAYOUTS["workflow"], workflow_request("example", "incident"), fields
        )
