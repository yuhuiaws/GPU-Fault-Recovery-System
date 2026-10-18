"""The store-maintenance script without a database.

``gpu-fault-admin store-maintenance`` execs this script's source into a CPU
ingress Pod (``python -c``, the request on stdin), and
``test_store_maintenance_script_postgres.py`` proves it against a real store.
That file runs only with a PostgreSQL grant, where administrator modules are
not measured, so the cases here drive every decision of the script through a
recorded catalog instead: what a plan reads and what its identity covers, each
refusal before the first maintenance statement, the per-table outcome report
and the lock release, the DSN sources, and the stdin/stdout program contract.
"""

from __future__ import annotations

import io
import json
import re
import runpy
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from gpu_fault.admin import store_maintenance_script as script
from gpu_fault.store.postgres import hot_table_bloat

if TYPE_CHECKING:
    # psycopg is the optional ``postgres`` extra: it is imported where a case
    # runs, never at module scope, or the whole suite fails to collect without
    # it (tests/store/test_store_override_contracts.py).
    from psycopg.pq import TransactionStatus

PAGE = 8192
SERVER_VERSION = "PostgreSQL 16.4 on aarch64-unknown-linux-gnu, compiled by gcc"
COMPACT = "VACUUM (FULL, ANALYZE)"
REINDEX = "REINDEX INDEX CONCURRENTLY + VACUUM (ANALYZE)"
HOT_TABLES = (*script.COMPACT_TABLES, *script.REINDEX_TABLES)
SIZE_KEYS = (
    "live_rows",
    "dead_rows",
    "heap_pages",
    "heap_bytes",
    "index_bytes",
    "toast_bytes",
    "total_bytes",
)
VACUUMED_AT = datetime(2026, 10, 17, 3, 0, tzinfo=UTC)
AUTOVACUUMED_AT = datetime(2026, 10, 17, 21, 30, 15, tzinfo=UTC)
# What psycopg sends for a composed maintenance statement: the verb, then the
# table or index name as a double-quoted identifier ("" inside escapes ").
MAINTENANCE = re.compile(
    r"(?P<verb>VACUUM \(FULL, ANALYZE\)|VACUUM \(ANALYZE\)|REINDEX INDEX CONCURRENTLY)"
    r' "(?P<name>(?:[^"]|"")+)"'
)


class LockNotAvailable(Exception):
    """What ``lock_timeout`` raises mid-statement; its text names a row."""


class QueryCanceled(Exception):
    """What ``statement_timeout`` raises mid-statement."""


class SessionLost(Exception):
    """Not a refusal: the store session went away under the script."""


@dataclass
class Table:
    """One hot table as the catalog reports it; maintenance shrinks it."""

    indexes: dict[str, int] = field(default_factory=dict)  # name -> pages
    invalid: set[str] = field(default_factory=set)
    live_rows: int = 40
    dead_rows: int = 260
    heap_pages: int = 30
    toast_bytes: int = 0
    reloptions: list[str] | None = field(default_factory=lambda: ["fillfactor=50"])
    toast_reloptions: list[str] | None = None
    updates: int = 9000
    hot_updates: int = 8800
    seq_scans: int = 3
    last_vacuum: datetime | None = None
    last_autovacuum: datetime | None = None

    def facts_row(self, name: str) -> tuple[Any, ...]:
        """The facts query's row; PostgreSQL ``sum`` of bigint is numeric,
        which psycopg returns as ``Decimal``."""

        heap_bytes = self.heap_pages * PAGE
        index_bytes = sum(self.indexes.values()) * PAGE
        return (
            name,
            self.heap_pages,
            Decimal(heap_bytes),
            Decimal(index_bytes),
            Decimal(self.toast_bytes),
            Decimal(heap_bytes + index_bytes + self.toast_bytes),
            self.reloptions,
            self.toast_reloptions,
            Decimal(self.live_rows),
            Decimal(self.dead_rows),
            Decimal(self.updates),
            Decimal(self.hot_updates),
            Decimal(self.seq_scans),
            self.last_vacuum,
            self.last_autovacuum,
        )

    def index_rows(self) -> list[tuple[str, Decimal, bool]]:
        return [
            (name, Decimal(pages * PAGE), name not in self.invalid)
            for name, pages in sorted(self.indexes.items())
        ]


class Store:
    """The catalog the script reads and the log of everything it ran.

    Catalog reads are logged by kind (``server``, ``facts <table>``,
    ``indexes <table>``, ``lock``, ``unlock``) with their bound parameters in
    ``bound``; maintenance statements by the SQL psycopg would send. Each
    statement advances ``clock`` by its entry in ``durations`` (0.5 s by
    default); ``failures`` raises at the n-th occurrence of a log entry and
    ``after`` runs once a maintenance statement succeeded.
    """

    def __init__(
        self,
        tables: dict[str, Table],
        *,
        version: str = SERVER_VERSION,
        version_num: int = 160004,
        lock_free: bool = True,
    ) -> None:
        self.tables = tables
        self.version = version
        self.version_num = version_num
        self.lock_free = lock_free
        self.log: list[str] = []
        self.bound: list[tuple[str, tuple[Any, ...]]] = []
        self.failures: dict[tuple[str, int], Exception] = {}
        self.after: dict[str, Callable[[], object]] = {}
        self.durations: dict[str, float] = {}
        self.clock = 0.0

    def monotonic(self) -> float:
        return self.clock

    def record(self, entry: str) -> None:
        self.log.append(entry)
        failure = self.failures.get((entry, self.log.count(entry)))
        if failure is not None:
            raise failure

    def answer(self, statement: Any, parameters: Any) -> list[tuple[Any, ...]]:
        from psycopg import sql

        if isinstance(statement, sql.Composable):
            return self.maintain(statement.as_string(None))
        bound = tuple(parameters or ())
        if "server_version_num" in statement:
            self.record("server")
            return [(self.version, self.version_num, "gpu_fault")]
        if "pg_stat_user_tables" in statement:
            self.bound.append(("facts", bound))
            self.record(f"facts {bound[0]}")
            table = self.tables.get(bound[0])
            return [] if table is None else [table.facts_row(bound[0])]
        if "FROM pg_index" in statement:
            self.bound.append(("indexes", bound))
            self.record(f"indexes {bound[0]}")
            return self.tables[bound[0]].index_rows()
        if "pg_try_advisory_lock" in statement:
            self.bound.append(("lock", bound))
            self.record("lock")
            return [(self.lock_free,)]
        if "pg_advisory_unlock" in statement:
            self.bound.append(("unlock", bound))
            self.record("unlock")
            return [(True,)]
        raise LookupError(f"not a statement the script sends: {statement}")

    def maintain(self, text: str) -> list[tuple[Any, ...]]:
        self.clock += self.durations.get(text, 0.5)
        self.record(text)
        match = MAINTENANCE.fullmatch(text)
        if match is None:
            raise LookupError(f"not a quoted maintenance statement: {text}")
        name = match["name"].replace('""', '"')
        if match["verb"] == "REINDEX INDEX CONCURRENTLY":
            (owner,) = [item for item in self.tables.values() if name in item.indexes]
            owner.indexes[name] = 1
        else:
            table = self.tables[name]
            table.dead_rows = 0
            if match["verb"] == "VACUUM (FULL, ANALYZE)":
                table.heap_pages = 1
                table.indexes = dict.fromkeys(table.indexes, 1)
        self.after.get(text, lambda: None)()
        return []


class Cursor:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.rows: list[tuple[Any, ...]] = []
        self.closed = False

    def __enter__(self) -> Cursor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closed = True

    def execute(self, statement: Any, parameters: Any = None) -> None:
        self.rows = self.store.answer(statement, parameters)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self.rows)


def transaction_status(name: str) -> TransactionStatus:
    """psycopg's own ``TransactionStatus`` member, imported when a case runs."""

    from psycopg.pq import TransactionStatus

    return TransactionStatus[name]


class Connection:
    """A psycopg connection as the script uses one; ``status`` names its
    ``TransactionStatus`` member."""

    def __init__(
        self, store: Store, *, autocommit: bool = True, status: str = "IDLE"
    ) -> None:
        self.store = store
        self.autocommit = autocommit
        self.info = SimpleNamespace(transaction_status=transaction_status(status))
        self.cursors: list[Cursor] = []
        self.session: list[str] = []
        self.closed = False

    def cursor(self) -> Cursor:
        cursor = Cursor(self.store)
        self.cursors.append(cursor)
        return cursor

    def execute(self, statement: str) -> None:
        self.session.append(statement)

    def close(self) -> None:
        self.closed = True


def hot_tables() -> dict[str, Table]:
    """Every hot table: the small ones with a primary key, the large ones with
    a second index and a day of churn."""

    tables = {
        name: Table(indexes={f"{name}_pkey": 4}) for name in script.COMPACT_TABLES
    }
    for name in script.REINDEX_TABLES:
        tables[name] = Table(
            indexes={f"{name}_pkey": 40, f"{name}_observed_idx": 24},
            live_rows=1200,
            dead_rows=5000,
            heap_pages=900,
        )
    return tables


def plan_reads(store: Store) -> list[str]:
    """What a plan reads: the server, then every hot table's facts in plan
    order, and the indexes of the tables that exist."""

    reads = ["server"]
    for name in HOT_TABLES:
        reads.append(f"facts {name}")
        if name in store.tables:
            reads.append(f"indexes {name}")
    return reads


def approve(store: Store) -> str:
    """The identity the operator reviewed: a plan of this very catalog. The
    reads are forgotten, so a case sees only what ``execute`` itself ran."""

    identity = script.plan(Cursor(store))["plan_sha256"]
    store.log.clear()
    store.bound.clear()
    return identity


def by_table(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {action["table"]: action for action in result["actions"]}


def install_store(
    monkeypatch: pytest.MonkeyPatch, store: Store
) -> list[SimpleNamespace]:
    """Route ``psycopg.connect`` to ``store``; return every connection opened."""

    opened: list[SimpleNamespace] = []

    def connect(dsn: str, **options: Any) -> Connection:
        connection = Connection(store)
        opened.append(SimpleNamespace(dsn=dsn, options=options, connection=connection))
        return connection

    monkeypatch.setattr("psycopg.connect", connect)
    return opened


def mount_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str = "postgresql://file\n"
) -> Path:
    path = tmp_path / "postgres-url"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    return path


def send(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """The request on stdin, the way ``kubectl exec -i`` delivers it."""

    monkeypatch.setattr(sys, "stdin", io.StringIO(text))


@pytest.fixture(autouse=True)
def default_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No credential reaches a case unless it mounts one: neither the host's
    environment nor its ``/etc/gpu-fault`` is a test input. Returns the
    (absent) default mount path."""

    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.delenv("GPU_FAULT_STORE_URL", raising=False)
    path = tmp_path / "default-mount" / "postgres-url"
    monkeypatch.setattr(script, "DEFAULT_STORE_URL_FILE", str(path))
    return path


# -- plan ------------------------------------------------------------------


def test_plan_compacts_only_the_small_tables_and_reindexes_the_large_ones() -> None:
    store = Store(hot_tables())

    plan = script.plan(Cursor(store))

    assert [(item["table"], item["action"]) for item in plan["tables"]] == [
        ("gpu_fault_processor_lanes", COMPACT),
        ("gpu_fault_processor_queue_counts", COMPACT),
        ("gpu_fault_processor_priority_count_shards", COMPACT),
        ("gpu_fault_gpu_metric_latest", REINDEX),
        ("gpu_fault_gpu_metric_latest_clusters", REINDEX),
        ("gpu_fault_objects", REINDEX),
        ("gpu_fault_processor_queue", REINDEX),
        ("gpu_fault_processor_payloads", REINDEX),
        ("gpu_fault_gpu_metrics_batches", REINDEX),
    ], "VACUUM FULL's exclusive lock is for the small in-place tables only"
    assert plan["missing_tables"] == []
    assert plan["server"] == {
        "version": SERVER_VERSION,
        "version_num": 160004,
        "database": "gpu_fault",
        "reindex_concurrently_supported": True,
    }
    assert re.fullmatch(r"[0-9a-f]{64}", plan["plan_sha256"]), plan["plan_sha256"]
    assert store.log == plan_reads(store), "a plan reads the catalog and runs nothing"


def test_plan_names_the_absent_tables_and_reads_no_indexes_for_them() -> None:
    tables = hot_tables()
    del tables["gpu_fault_processor_queue_counts"]
    del tables["gpu_fault_gpu_metric_latest_clusters"]
    store = Store(tables)

    plan = script.plan(Cursor(store))

    assert plan["missing_tables"] == [
        "gpu_fault_processor_queue_counts",
        "gpu_fault_gpu_metric_latest_clusters",
    ]
    assert [item["table"] for item in plan["tables"]] == [
        name for name in HOT_TABLES if name in tables
    ]
    assert "indexes gpu_fault_processor_queue_counts" not in store.log
    assert store.log == plan_reads(store), "absent tables are read once, not indexed"


def test_table_facts_turn_one_catalog_row_into_a_whole_number_report() -> None:
    store = Store(
        {
            "gpu_fault_objects": Table(
                indexes={
                    "gpu_fault_objects_pkey": 40,
                    "gpu_fault_objects_kind_idx": 24,
                    "gpu_fault_objects_kind_idx_ccnew": 3,
                },
                invalid={"gpu_fault_objects_kind_idx_ccnew"},
                live_rows=1200,
                dead_rows=5000,
                heap_pages=900,
                toast_bytes=3 * PAGE,
                reloptions=[
                    "toast_tuple_target=8160",
                    "fillfactor=70",
                    "autovacuum_vacuum_threshold=500",
                ],
                toast_reloptions=["autovacuum_vacuum_threshold=500"],
                updates=91000,
                hot_updates=88000,
                seq_scans=7,
                last_vacuum=VACUUMED_AT,
                last_autovacuum=AUTOVACUUMED_AT,
            )
        }
    )

    facts = script.table_facts(Cursor(store), "gpu_fault_objects")

    assert facts == {
        "table": "gpu_fault_objects",
        "live_rows": 1200,
        "dead_rows": 5000,
        "heap_pages": 900,
        "heap_bytes": 900 * PAGE,
        "index_bytes": 67 * PAGE,
        "toast_bytes": 3 * PAGE,
        "total_bytes": 970 * PAGE,
        "reloptions": [
            "autovacuum_vacuum_threshold=500",
            "fillfactor=70",
            "toast_tuple_target=8160",
        ],
        "toast_reloptions": ["autovacuum_vacuum_threshold=500"],
        "updates": 91000,
        "hot_updates": 88000,
        "seq_scans": 7,
        "last_vacuum": "2026-10-17T03:00:00+00:00",
        "last_autovacuum": "2026-10-17T21:30:15+00:00",
        "indexes": [
            {"name": "gpu_fault_objects_kind_idx", "bytes": 24 * PAGE, "valid": True},
            {
                "name": "gpu_fault_objects_kind_idx_ccnew",
                "bytes": 3 * PAGE,
                "valid": False,
            },
            {"name": "gpu_fault_objects_pkey", "bytes": 40 * PAGE, "valid": True},
        ],
    }
    # Decimal(5000) == 5000, so the equality above cannot see the conversion.
    counted = (*SIZE_KEYS, "updates", "hot_updates", "seq_scans")
    assert {key: type(facts[key]) for key in counted} == dict.fromkeys(counted, int), (
        "numeric sums leave the script as whole numbers"
    )
    assert json.loads(json.dumps(facts)) == facts, "the report is plain JSON"
    assert store.bound == [
        ("facts", ("gpu_fault_objects",) * 7),
        ("indexes", ("gpu_fault_objects",)),
    ], "every placeholder of both reads binds the table name"


def test_table_facts_of_a_table_without_options_vacuums_or_indexes() -> None:
    store = Store({"gpu_fault_processor_lanes": Table(reloptions=None)})

    facts = script.table_facts(Cursor(store), "gpu_fault_processor_lanes")

    assert facts is not None, "the table exists"
    unset = (
        "reloptions",
        "toast_reloptions",
        "indexes",
        "last_vacuum",
        "last_autovacuum",
    )
    assert {key: facts[key] for key in unset} == {
        "reloptions": [],
        "toast_reloptions": [],
        "indexes": [],
        "last_vacuum": None,
        "last_autovacuum": None,
    }, "NULL catalog columns read as empty lists and absent timestamps"


def test_table_facts_of_an_absent_table_are_none_without_an_index_read() -> None:
    store = Store({})

    assert script.table_facts(Cursor(store), "gpu_fault_processor_queue") is None
    assert store.log == ["facts gpu_fault_processor_queue"]


@pytest.mark.parametrize(
    ("version_num", "supported"),
    [(110022, False), (119999, False), (120000, True), (160004, True)],
)
def test_server_facts_gate_reindex_concurrently_on_postgresql_12(
    version_num: int, supported: bool
) -> None:
    store = Store({}, version="PostgreSQL", version_num=version_num)

    assert script.server_facts(Cursor(store)) == {
        "version": "PostgreSQL",
        "version_num": version_num,
        "database": "gpu_fault",
        "reindex_concurrently_supported": supported,
    }
    assert script.reindex_concurrently_supported(version_num) is supported


def test_the_plan_identity_names_tables_actions_and_indexes_not_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def identity(tables: dict[str, Table]) -> str:
        return script.plan(Cursor(Store(tables)))["plan_sha256"]

    catalog = hot_tables()
    reviewed = identity(catalog)

    churned = hot_tables()
    for table in churned.values():
        table.live_rows += 7
        table.dead_rows *= 10
        table.heap_pages += 500
        table.indexes = {name: pages * 3 for name, pages in table.indexes.items()}
        table.last_autovacuum = AUTOVACUUMED_AT
    assert identity(churned) == reviewed, (
        "sizes move between the review and the execute; the identity must not"
    )

    indexed = hot_tables()
    indexed["gpu_fault_objects"].indexes["gpu_fault_objects_owner_idx"] = 2
    assert identity(indexed) != reviewed, "a new index is a statement nobody reviewed"

    shrunk = hot_tables()
    del shrunk["gpu_fault_processor_lanes"]
    assert identity(shrunk) != reviewed, "a table that went away is another plan"

    first = script.REINDEX_TABLES[0]
    monkeypatch.setattr(script, "COMPACT_TABLES", (*script.COMPACT_TABLES, first))
    monkeypatch.setattr(script, "REINDEX_TABLES", script.REINDEX_TABLES[1:])
    assert identity(catalog) != reviewed, (
        "the same tables and indexes in the same order under another action"
    )


# -- execute: refusals before the first maintenance statement --------------


@pytest.mark.parametrize(
    "session",
    [
        pytest.param(
            lambda store: Connection(store, autocommit=False), id="no-autocommit"
        ),
        pytest.param(
            lambda store: Connection(store, status="INTRANS"), id="inside-a-transaction"
        ),
        pytest.param(
            lambda store: Connection(store, status="INERROR"), id="failed-transaction"
        ),
        pytest.param(
            lambda store: SimpleNamespace(
                info=SimpleNamespace(transaction_status=transaction_status("IDLE"))
            ),
            id="no-autocommit-attribute",
        ),
    ],
)
def test_execute_needs_an_idle_autocommit_session(
    session: Callable[[Store], Any],
) -> None:
    store = Store(hot_tables())
    connection = session(store)

    with pytest.raises(
        script.MaintenanceRefused,
        match="^VACUUM and REINDEX CONCURRENTLY need an idle autocommit connection$",
    ):
        script.execute(connection, "0" * 64)

    assert getattr(connection, "cursors", []) == [], "refused before a cursor opens"
    assert store.log == [], "nothing was read or run"


def test_execute_replans_and_refuses_a_store_that_changed_since_the_review() -> None:
    store = Store(hot_tables())
    approved = approve(store)
    store.tables["gpu_fault_objects"].indexes["gpu_fault_objects_owner_idx"] = 2

    with pytest.raises(
        script.MaintenanceRefused,
        match="^the store changed since the plan was reviewed; plan again$",
    ):
        script.execute(Connection(store), approved)

    assert store.log == plan_reads(store), "re-planned, then refused before the lock"


def test_execute_refuses_postgresql_before_12_and_names_the_server() -> None:
    old = "PostgreSQL 11.22 on x86_64-pc-linux-gnu, compiled by gcc"
    store = Store(hot_tables(), version=old, version_num=110022)
    approved = approve(store)

    with pytest.raises(script.MaintenanceRefused) as refused:
        script.execute(Connection(store), approved)

    assert str(refused.value) == (
        f"REINDEX INDEX CONCURRENTLY needs PostgreSQL 12 or newer; the store runs {old}"
    )
    assert store.log == plan_reads(store), "refused before the lock"


def test_execute_refuses_while_the_lock_is_held_and_never_releases_it() -> None:
    store = Store(hot_tables(), lock_free=False)
    approved = approve(store)

    with pytest.raises(
        script.MaintenanceRefused,
        match="^a schema ensure or index build holds the store maintenance lock$",
    ):
        script.execute(Connection(store), approved)

    assert store.log == [*plan_reads(store), "lock"], (
        "no maintenance statement, and no unlock of a lock this session never took"
    )


# -- execute: the run ------------------------------------------------------


def test_execute_compacts_reindexes_and_reports_each_table_under_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store(
        {
            "gpu_fault_processor_lanes": Table(
                indexes={"gpu_fault_processor_lanes_pkey": 4}
            ),
            "gpu_fault_objects": Table(
                indexes={
                    "gpu_fault_objects_pkey": 40,
                    "gpu_fault_objects_kind_idx": 24,
                },
                live_rows=1200,
                dead_rows=5000,
                heap_pages=900,
            ),
        }
    )
    store.durations = {
        'VACUUM (FULL, ANALYZE) "gpu_fault_processor_lanes"': 0.2104,
        'REINDEX INDEX CONCURRENTLY "gpu_fault_objects_kind_idx"': 3.5,
        'REINDEX INDEX CONCURRENTLY "gpu_fault_objects_pkey"': 12.3456,
    }
    monkeypatch.setattr(script, "time", SimpleNamespace(monotonic=store.monotonic))
    approved = approve(store)
    replanned = plan_reads(store)

    result = script.execute(Connection(store), approved)

    assert store.log == [
        *replanned,
        "lock",
        'VACUUM (FULL, ANALYZE) "gpu_fault_processor_lanes"',
        "facts gpu_fault_processor_lanes",
        "indexes gpu_fault_processor_lanes",
        'REINDEX INDEX CONCURRENTLY "gpu_fault_objects_kind_idx"',
        'REINDEX INDEX CONCURRENTLY "gpu_fault_objects_pkey"',
        'VACUUM (ANALYZE) "gpu_fault_objects"',
        "facts gpu_fault_objects",
        "indexes gpu_fault_objects",
        "unlock",
    ], "re-plan, lock, each table's statements then its after-read, unlock"
    assert (result["status"], result["errors"]) == ("SUCCEEDED", [])
    assert result["plan_sha256"] == approved
    assert result["server"]["version"] == SERVER_VERSION
    lanes, objects = result["actions"]
    assert lanes == {
        "table": "gpu_fault_processor_lanes",
        "action": COMPACT,
        "status": "SUCCEEDED",
        "steps": [
            {
                "statement": "VACUUM (FULL, ANALYZE) gpu_fault_processor_lanes",
                "seconds": 0.21,
            }
        ],
        "before": {
            "live_rows": 40,
            "dead_rows": 260,
            "heap_pages": 30,
            "heap_bytes": 30 * PAGE,
            "index_bytes": 4 * PAGE,
            "toast_bytes": 0,
            "total_bytes": 34 * PAGE,
        },
        "after": {
            "live_rows": 40,
            "dead_rows": 0,
            "heap_pages": 1,
            "heap_bytes": PAGE,
            "index_bytes": PAGE,
            "toast_bytes": 0,
            "total_bytes": 2 * PAGE,
        },
    }
    assert (objects["table"], objects["action"], objects["status"]) == (
        "gpu_fault_objects",
        REINDEX,
        "SUCCEEDED",
    )
    assert objects["steps"] == [
        {
            "statement": "REINDEX INDEX CONCURRENTLY gpu_fault_objects_kind_idx",
            "seconds": 3.5,
        },
        {
            "statement": "REINDEX INDEX CONCURRENTLY gpu_fault_objects_pkey",
            "seconds": 12.346,
        },
        {"statement": "VACUUM (ANALYZE) gpu_fault_objects", "seconds": 0.5},
    ], "every index in catalog order, then the vacuum; each step timed on its own"
    assert (objects["before"]["index_bytes"], objects["after"]["index_bytes"]) == (
        64 * PAGE,
        2 * PAGE,
    )
    assert (objects["before"]["heap_pages"], objects["after"]["heap_pages"]) == (
        900,
        900,
    ), "plain VACUUM returns no heap pages; only the indexes shrink"
    assert set(objects["before"]) == set(SIZE_KEYS) == set(objects["after"])


def test_the_lock_is_the_one_the_schema_ensure_and_the_compactor_take() -> None:
    store = Store(hot_tables())

    script.execute(Connection(store), approve(store))

    assert [entry for entry in store.bound if entry[0] in {"lock", "unlock"}] == [
        ("lock", ("gpu_fault_schema_bootstrap",)),
        ("unlock", ("gpu_fault_schema_bootstrap",)),
    ]
    assert script.ADVISORY_LOCK_NAME == hot_table_bloat.ADVISORY_LOCK_NAME, (
        "the operator verb and the background compactor must exclude each other"
    )


def test_execute_sends_every_name_as_a_quoted_identifier() -> None:
    odd = 'gpu_fault_objects_Kind"idx'
    store = Store(
        {
            "gpu_fault_processor_lanes": Table(),
            "gpu_fault_objects": Table(indexes={odd: 3}),
        }
    )

    result = script.execute(Connection(store), approve(store))

    assert result["status"] == "SUCCEEDED", result["errors"]
    assert 'REINDEX INDEX CONCURRENTLY "gpu_fault_objects_Kind""idx"' in store.log, (
        store.log
    )
    assert 'VACUUM (FULL, ANALYZE) "gpu_fault_processor_lanes"' in store.log
    assert by_table(result)["gpu_fault_objects"]["steps"][0]["statement"] == (
        f"REINDEX INDEX CONCURRENTLY {odd}"
    ), "the report names the index as the catalog spells it"


def test_a_failed_table_is_reported_by_error_type_and_the_rest_still_run() -> None:
    store = Store(hot_tables())
    approved = approve(store)
    leaked = "canceling statement due to lock timeout on row node-7/gpu-3"
    store.failures[('VACUUM (FULL, ANALYZE) "gpu_fault_processor_queue_counts"', 1)] = (
        QueryCanceled("canceling statement due to statement timeout")
    )
    store.failures[('REINDEX INDEX CONCURRENTLY "gpu_fault_objects_pkey"', 1)] = (
        LockNotAvailable(leaked)
    )

    result = script.execute(Connection(store), approved)

    assert result["status"] == "FAILED"
    assert result["errors"] == [
        {"table": "gpu_fault_processor_queue_counts", "error": "QueryCanceled"},
        {"table": "gpu_fault_objects", "error": "LockNotAvailable"},
    ], "every failed table, in plan order, by error type"
    assert leaked not in json.dumps(result), "error text can carry record contents"
    actions = by_table(result)
    assert {name: action["status"] for name, action in actions.items()} == {
        name: (
            "FAILED"
            if name in {"gpu_fault_processor_queue_counts", "gpu_fault_objects"}
            else "SUCCEEDED"
        )
        for name in HOT_TABLES
    }
    counts = actions["gpu_fault_processor_queue_counts"]
    assert counts["steps"] == []
    assert counts["after"] == counts["before"], "the failed rewrite changed nothing"
    objects = actions["gpu_fault_objects"]
    assert [step["statement"] for step in objects["steps"]] == [
        "REINDEX INDEX CONCURRENTLY gpu_fault_objects_observed_idx"
    ], "the steps that completed before the failure"
    assert 'VACUUM (ANALYZE) "gpu_fault_objects"' not in store.log
    assert objects["after"]["index_bytes"] == 41 * PAGE, "the rebuilt index is measured"
    assert store.log.index('VACUUM (ANALYZE) "gpu_fault_gpu_metrics_batches"') > (
        store.log.index('REINDEX INDEX CONCURRENTLY "gpu_fault_objects_pkey"')
    ), "later tables still run"
    assert store.log[-1] == "unlock"


def test_a_table_gone_by_its_after_read_reports_no_after_sizes() -> None:
    store = Store(hot_tables())
    approved = approve(store)
    store.after['VACUUM (ANALYZE) "gpu_fault_processor_payloads"'] = lambda: (
        store.tables.pop("gpu_fault_processor_payloads")
    )

    result = script.execute(Connection(store), approved)

    payloads = by_table(result)["gpu_fault_processor_payloads"]
    assert (payloads["status"], payloads["after"]) == ("SUCCEEDED", None)
    assert set(payloads["before"]) == set(SIZE_KEYS)
    assert result["status"] == "SUCCEEDED"
    assert store.log[-1] == "unlock"


def test_the_lock_is_released_when_the_session_fails_mid_run() -> None:
    store = Store(hot_tables())
    approved = approve(store)
    # Its first read is the re-plan; the second is the after-read.
    store.failures[("facts gpu_fault_processor_queue_counts", 2)] = SessionLost(
        "server closed the connection unexpectedly"
    )

    with pytest.raises(SessionLost):
        script.execute(Connection(store), approved)

    assert store.log[-2:] == ["facts gpu_fault_processor_queue_counts", "unlock"], (
        "the advisory lock is released on the way out"
    )
    assert not [
        entry
        for entry in store.log
        if entry.endswith('"gpu_fault_processor_priority_count_shards"')
    ], "no table after the failure is touched"


# -- run: the request modes ------------------------------------------------


def test_run_plans_through_one_cursor_and_labels_the_mode() -> None:
    store = Store(hot_tables())
    connection = Connection(store)

    result = script.run({"mode": "plan"}, connection)

    assert result == {"mode": "plan", **script.plan(Cursor(Store(hot_tables())))}
    assert [cursor.closed for cursor in connection.cursors] == [True]
    assert store.log == plan_reads(store), "plan mode runs nothing"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"mode": "execute"}, id="absent"),
        pytest.param({"mode": "execute", "plan_sha256": None}, id="null"),
        pytest.param({"mode": "execute", "plan_sha256": ""}, id="empty"),
        pytest.param({"mode": "execute", "plan_sha256": "a" * 63}, id="short"),
        pytest.param({"mode": "execute", "plan_sha256": "a" * 65}, id="long"),
    ],
)
def test_run_executes_only_a_64_character_reviewed_identity(
    payload: dict[str, Any],
) -> None:
    store = Store(hot_tables())
    connection = Connection(store)

    with pytest.raises(
        script.MaintenanceRefused, match="^execute requires the reviewed plan identity$"
    ):
        script.run(payload, connection)

    assert connection.cursors == [], "refused before the store is read"


def test_run_dispatches_an_execute_with_the_reviewed_identity() -> None:
    store = Store(hot_tables())
    approved = approve(store)

    result = script.run({"mode": "execute", "plan_sha256": approved}, Connection(store))

    assert set(result) == {
        "mode",
        "server",
        "plan_sha256",
        "actions",
        "errors",
        "status",
    }
    assert (result["mode"], result["status"]) == ("execute", "SUCCEEDED")
    assert [action["table"] for action in result["actions"]] == list(HOT_TABLES)


@pytest.mark.parametrize(
    ("payload", "named"),
    [
        ({"mode": "vacuum-everything"}, "'vacuum-everything'"),
        ({"mode": "PLAN"}, "'PLAN'"),
        ({}, "None"),
    ],
)
def test_run_refuses_any_other_mode_by_name(
    payload: dict[str, Any], named: str
) -> None:
    connection = Connection(Store(hot_tables()))

    with pytest.raises(script.MaintenanceRefused) as refused:
        script.run(payload, connection)

    assert str(refused.value) == f"unsupported store maintenance mode: {named}"
    assert connection.cursors == [], "refused before the store is read"


# -- the store connection --------------------------------------------------


def test_store_dsn_prefers_the_named_secret_file_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount_secret(tmp_path, monkeypatch, "  postgresql://file\n")
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://env")

    assert script.store_dsn() == "postgresql://file"


def test_store_dsn_reads_the_default_mount_when_no_file_is_named(
    default_secret: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_secret.parent.mkdir()
    default_secret.write_text("postgresql://mounted\n", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", "   ")
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://env")

    assert script.store_dsn() == "postgresql://mounted", "a blank file name is unset"


def test_an_empty_secret_file_falls_through_to_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount_secret(tmp_path, monkeypatch, "\n")
    monkeypatch.setenv("GPU_FAULT_STORE_URL", " postgresql://env \n")

    assert script.store_dsn() == "postgresql://env"


def test_store_dsn_refuses_when_no_credential_is_mounted_or_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "  ")

    with pytest.raises(
        script.MaintenanceRefused,
        match="^store credentials are unavailable in this Pod$",
    ):
        script.store_dsn()


def test_connect_opens_an_autocommit_session_with_bounded_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = install_store(monkeypatch, Store({}))

    connection = script.connect("postgresql://pod")

    assert [(item.dsn, item.options) for item in opened] == [
        ("postgresql://pod", {"autocommit": True, "connect_timeout": 10})
    ]
    assert connection is opened[0].connection
    assert connection.session == [
        "SET lock_timeout = '10s'",
        "SET statement_timeout = '840s'",
    ], "no maintenance statement waits unbounded on a lock or runs unbounded"


# -- main: the program the verb execs into the Pod ------------------------


def test_main_answers_one_sorted_json_document_and_closes_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened = install_store(monkeypatch, Store(hot_tables()))
    mount_secret(tmp_path, monkeypatch)
    send(monkeypatch, json.dumps({"mode": "plan"}))

    assert script.main() == 0

    printed = capsys.readouterr().out
    result = json.loads(printed)
    assert printed == json.dumps(result, sort_keys=True) + "\n", "one sorted line"
    assert result["mode"] == "plan"
    assert [item["table"] for item in result["tables"]] == list(HOT_TABLES)
    assert [(item.dsn, item.connection.closed) for item in opened] == [
        ("postgresql://file", True)
    ]
    assert opened[0].connection.session == [
        "SET lock_timeout = '10s'",
        "SET statement_timeout = '840s'",
    ]


def test_main_prints_a_refusal_and_exits_1_after_closing_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store(hot_tables())
    opened = install_store(monkeypatch, store)
    mount_secret(tmp_path, monkeypatch)
    send(monkeypatch, json.dumps({"mode": "execute", "plan_sha256": "abc"}))

    assert script.main() == 1

    assert json.loads(capsys.readouterr().out) == {
        "error": "execute requires the reviewed plan identity",
        "error_kind": "refused",
    }
    assert [item.connection.closed for item in opened] == [True]
    assert store.log == [], "refused before the store is read"


def test_main_refuses_without_credentials_and_never_connects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened = install_store(monkeypatch, Store(hot_tables()))
    send(monkeypatch, json.dumps({"mode": "plan"}))

    assert script.main() == 1

    assert json.loads(capsys.readouterr().out) == {
        "error": "store credentials are unavailable in this Pod",
        "error_kind": "refused",
    }
    assert opened == [], "no connection without a DSN"


@pytest.mark.parametrize("text", ["[]", "null", '"plan"', "7"])
def test_main_refuses_a_request_that_is_not_an_object_before_connecting(
    text: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opened = install_store(monkeypatch, Store(hot_tables()))
    mount_secret(tmp_path, monkeypatch)
    send(monkeypatch, text)

    with pytest.raises(
        script.MaintenanceRefused, match="^store maintenance request is not an object$"
    ):
        script.main()

    assert opened == [], "the request is checked before the store is reached"
    assert capsys.readouterr().out == ""


def test_main_closes_the_session_when_the_store_fails_and_prints_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store(hot_tables())
    store.failures[("server", 1)] = SessionLost(
        "server closed the connection unexpectedly"
    )
    opened = install_store(monkeypatch, store)
    mount_secret(tmp_path, monkeypatch)
    send(monkeypatch, json.dumps({"mode": "plan"}))

    with pytest.raises(SessionLost):
        script.main()

    assert [item.connection.closed for item in opened] == [True]
    assert capsys.readouterr().out == "", (
        "no half result: the Pod exec fails with the traceback instead"
    )


def test_the_module_is_the_pod_program_and_exits_with_mains_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``python -c <source>`` runs the module as ``__main__``: its exit status
    is ``main``'s answer, 0 for a result and 1 for a refusal."""

    install_store(monkeypatch, Store(hot_tables()))
    mount_secret(tmp_path, monkeypatch)

    send(monkeypatch, json.dumps({"mode": "plan"}))
    with pytest.raises(SystemExit) as planned:
        runpy.run_path(script.__file__, run_name="__main__")
    assert planned.value.code == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "plan"

    send(monkeypatch, json.dumps({"mode": "vacuum-everything"}))
    with pytest.raises(SystemExit) as refused:
        runpy.run_path(script.__file__, run_name="__main__")
    assert refused.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "unsupported store maintenance mode: 'vacuum-everything'",
        "error_kind": "refused",
    }
