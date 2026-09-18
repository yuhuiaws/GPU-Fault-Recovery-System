from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from typing import Any

import pytest

from gpu_fault.installation_inventory import installed_unit_report
from tests._script_loader import lazy_script_module
from tests.deploy._cov95_tools_support import TOOLS

ACTIVITY = lazy_script_module(TOOLS / "cleanup_activity.py")


class Cursor:
    def __init__(self, rows: list[Any], all_rows: list[Any] | None = None) -> None:
        self.rows = iter(rows)
        self.all_rows = all_rows or []
        self.calls: list[tuple[str, tuple[Any, ...] | None]] = []

    def execute(
        self, statement: Any, parameters: tuple[Any, ...] | None = None
    ) -> None:
        text = statement if isinstance(statement, str) else statement.as_string()
        self.calls.append((text, parameters))

    def fetchone(self) -> Any:
        return next(self.rows)

    def fetchall(self) -> list[Any]:
        return self.all_rows

    def __enter__(self) -> Cursor:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class Connection:
    def __init__(self, cursor: Cursor) -> None:
        self.reader = cursor
        self.transactions: list[str] = []

    def cursor(self) -> Cursor:
        return self.reader

    @contextmanager
    def transaction(self):
        self.transactions.append("begin")
        try:
            yield
        except Exception:
            self.transactions.append("rollback")
            raise
        else:
            self.transactions.append("commit")

    def __enter__(self) -> Connection:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


@pytest.mark.parametrize(
    "relations,modes,expected",
    [
        ((None, None, None, None), [], "gpu_fault_objects"),
        (("records", None, None, None), [], "gpu_fault_control_records"),
        (
            ("records", "modes", "commands", "workflows"),
            [("workflow", "dedicated"), ("remote_command", "dual")],
            "gpu_fault_control_records",
        ),
    ],
)
def test_cleanup_count_relation_follows_proven_storage_mode(
    relations: tuple[Any, ...], modes: list[Any], expected: str
) -> None:
    cursor = Cursor([relations], modes)
    assert ACTIVITY.records_relation(cursor) == expected
    assert len(cursor.calls) == (2 if relations[1] else 1)


@pytest.mark.parametrize(
    "relations,modes,problem",
    [
        (
            (None, "modes", "commands", "workflows"),
            [("workflow", "legacy"), ("remote_command", "legacy")],
            "not known",
        ),
        (("records", "modes", None, "workflows"), [], "not known"),
        (
            ("records", "modes", "commands", "workflows"),
            [("workflow", "unknown"), ("remote_command", "legacy")],
            "not known",
        ),
        (("records", None, "commands", None), [], "metadata is missing"),
        (("records", None, None, "workflows"), [], "metadata is missing"),
    ],
)
def test_cleanup_refuses_partial_or_unknown_storage_metadata(
    relations: tuple[Any, ...], modes: list[Any], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        ACTIVITY.records_relation(Cursor([relations], modes))


@pytest.mark.parametrize("scope", ["all", "gpu"])
@pytest.mark.parametrize("spool", [False, True])
def test_counts_use_repeatable_read_and_bind_selected_clusters(
    scope: str, spool: bool
) -> None:
    cursor = Cursor(
        [
            (None, None, None, None),
            (2,),
            (3,),
            (4,),
            ("spool" if spool else None,),
            (5,),
        ]
    )
    connection = Connection(cursor)
    assert ACTIVITY.counts(connection, scope=scope, cluster_ids=["gpu-a"]) == (
        2,
        3,
        4,
        5 if spool else 0,
    )
    assert connection.transactions == ["begin", "commit"]
    assert cursor.calls[:2] == [
        ("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY", None),
        ("SET LOCAL row_security = off", None),
    ]
    count_queries = [
        (text, parameters) for text, parameters in cursor.calls if "count(*)" in text
    ]
    assert len(count_queries) == 3 + int(spool)
    assert all(
        parameters == ((["gpu-a"],) if scope == "gpu" else ())
        for _, parameters in count_queries
    ), "cleanup counts lost the selected cluster parameter binding"
    assert all(("ANY(%s)" in text) is (scope == "gpu") for text, _ in count_queries), (
        "scoped cleanup counts must constrain every queried activity table"
    )
    assert "LEFT JOIN" in count_queries[0][0], "orphan workflows must remain blockers"


@pytest.mark.parametrize("scope,cluster_ids", [("unknown", []), ("gpu", [])])
def test_invalid_query_scope_starts_no_transaction(
    scope: str, cluster_ids: list[str]
) -> None:
    connection = Connection(Cursor([]))
    with pytest.raises(ValueError, match="invalid cleanup query scope"):
        ACTIVITY.counts(connection, scope=scope, cluster_ids=cluster_ids)
    assert connection.transactions == []


@pytest.mark.parametrize("bad", [None, (-1,), ("0",), (True,)])
@pytest.mark.parametrize("spool", [False, True])
def test_invalid_database_counts_fail_without_fabricating_zero(
    bad: Any, spool: bool
) -> None:
    rows = (
        [(None, None, None, None), (0,), (0,), (0,), ("spool",), bad]
        if spool
        else [(None, None, None, None), bad]
    )
    connection = Connection(Cursor(rows))
    with pytest.raises(ValueError, match="invalid .*count"):
        ACTIVITY.counts(connection, scope="all", cluster_ids=[])
    assert connection.transactions == ["begin", "rollback"]


def test_enabled_missing_spool_is_not_zero_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL_ENABLED", "true")
    connection = Connection(
        Cursor([(None, None, None, None), (0,), (0,), (0,), (None,)])
    )
    with pytest.raises(ValueError, match="spool table is missing"):
        ACTIVITY.counts(connection, scope="all", cluster_ids=[])
    assert connection.transactions[-1] == "rollback"


@pytest.mark.parametrize(
    "inventory",
    [
        None,
        installed_unit_report(
            ["gpu-fault-node-agent.service"], include_units=True
        ).model_dump(mode="json"),
    ],
)
def test_fleet_inventory_is_scoped_and_contains_no_uncaptured_agent_fields(
    inventory: Any,
) -> None:
    agent = {
        "cluster_id": "gpu-a",
        "node_id": "node-a",
        "node_instance_id": "i-example",
        "lifecycle_state": "ACTIVE",
        "installed_unit_inventory": inventory,
        "extra_private_field": "not exported",
    }
    connection = Connection(Cursor([], [(agent,)]))
    result = ACTIVITY.fleet_inventory(connection, ["gpu-a"])
    assert result == [
        {
            name: agent[name]
            for name in (
                "cluster_id",
                "node_id",
                "node_instance_id",
                "lifecycle_state",
                "installed_unit_inventory",
            )
        }
    ]
    assert connection.reader.calls[0][1] == (["gpu-a"],)


@pytest.mark.parametrize(
    "rows",
    [
        [({"cluster_id": "other", "node_id": "node-a"},)],
        [({"cluster_id": "gpu-a", "node_id": ""},)],
        [({"cluster_id": "gpu-a", "node_id": 1},)],
        [({"cluster_id": "gpu-a", "node_id": "node-a"},)] * 2,
    ],
)
def test_fleet_identity_must_be_unique_and_within_selected_cluster(
    rows: list[Any],
) -> None:
    with pytest.raises(ValueError, match="ambiguous fleet identity"):
        ACTIVITY.fleet_inventory(Connection(Cursor([], rows)), ["gpu-a"])


def test_oversized_fleet_inventory_is_refused_before_partial_output() -> None:
    rows = [({"cluster_id": "gpu-a", "node_id": "node-a"},)] * 100001
    with pytest.raises(ValueError, match="exceeds cleanup limit"):
        ACTIVITY.fleet_inventory(Connection(Cursor([], rows)), ["gpu-a"])


@pytest.mark.parametrize("action", ["fleet", "counts", "unknown"])
def test_probe_entrypoint_uses_fake_database_and_prints_only_public_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], action: str
) -> None:
    import psycopg

    monkeypatch.setenv(
        "GPU_FAULT_STORE_URL",
        "host=example.invalid sslmode=verify-full sslrootcert=/example/ca",
    )
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setattr(sys, "argv", ["cleanup_activity", action, "all", "gpu-a"])
    connection = Connection(
        Cursor([(None, None, None, None), (0,), (0,), (0,), (None,)])
    )
    observed = []

    def connect(**kwargs: Any) -> Connection:
        observed.append(kwargs)
        return connection

    monkeypatch.setattr(psycopg, "connect", connect)
    assert ACTIVITY.main() == (1 if action == "unknown" else 0)
    output = capsys.readouterr()
    if action == "fleet":
        assert json.loads(output.out) == []
    elif action == "counts":
        assert output.out.strip() == "0\t0\t0\t0"
    else:
        assert output.err.strip() == "cleanup database probe failed: ValueError"
        assert output.out == ""
    assert observed[0]["connect_timeout"] == 10
    assert observed[0]["sslmode"] == "verify-full"


def test_missing_database_configuration_is_reported_without_opening_connection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import psycopg

    monkeypatch.delenv("GPU_FAULT_STORE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setattr(sys, "argv", ["cleanup_activity", "counts", "all"])

    def forbidden(**_kwargs: Any) -> None:
        pytest.fail("missing credentials must not open any database connection")

    monkeypatch.setattr(psycopg, "connect", forbidden)
    assert ACTIVITY.main() == 1
    assert (
        capsys.readouterr().err.strip() == "cleanup database probe failed: ValueError"
    )
