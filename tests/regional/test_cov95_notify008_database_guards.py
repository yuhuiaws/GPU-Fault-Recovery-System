from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.e2e.regional.probes import notify008_postgres as module
from scripts.e2e.regional.probes.notify008_protocol import ProbeError
from tests.regional._cov95_notify008_support import RUN_ID

IDENTITY = ("notify008", "notify008", None, "", "/database/data")


class Connection:
    def __init__(
        self,
        *,
        version=160001,
        identity=IDENTITY,
        aurora=(False,),
        tables=None,
        owner=None,
    ):
        self.info = SimpleNamespace(server_version=version)
        self.identity, self.aurora = identity, aurora
        self.tables = [] if tables is None else tables
        self.owner = [(RUN_ID,)] if owner is None else owner
        self.commands = []
        self.answer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, parameters=None):
        self.commands.append((query, parameters))
        if "current_database()" in query:
            self.answer = self.identity
        elif "aurora_version" in query:
            self.answer = self.aurora
        elif "pg_tables" in query:
            self.answer = self.tables
        elif query.startswith("SELECT run_id"):
            self.answer = self.owner
        return self

    def fetchone(self):
        return self.answer

    def fetchall(self):
        return self.answer


def test_fixed_database_connection_has_no_inherited_target(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "psycopg.connect",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "connection",
    )
    assert module.connect() == "connection", (
        "the fixed local target must be passed to the transport"
    )
    args, kwargs = calls[0]
    assert args == (module.DSN,), "the caller must not select another database endpoint"
    assert kwargs["connect_timeout"] == 5, (
        "connection establishment must remain bounded"
    )
    assert "statement_timeout=5000" in kwargs["options"], (
        "database statements must be bounded"
    )


@pytest.mark.parametrize(
    "name", ["PGSERVICE", "PGHOSTADDR", "PGPASSWORD", "GPU_FAULT_STORE_URL_FILE"]
)
def test_ambient_database_targets_are_refused_before_connection(monkeypatch, name):
    monkeypatch.setenv(name, "unapproved")
    monkeypatch.setattr(
        "psycopg.connect",
        lambda *args, **kwargs: pytest.fail("ambient target reached PostgreSQL"),
    )
    with pytest.raises(ProbeError, match="ambient"):
        module.connect()


@pytest.mark.parametrize("version", [True, 150099, 170000, "160001"])
def test_backend_must_be_observed_postgresql16(version):
    connection = Connection(version=version)
    with pytest.raises(ProbeError, match="version 16"):
        module.database_identity(connection)
    assert connection.commands == [], (
        "an incompatible backend must not reach fixture SQL"
    )


@pytest.mark.parametrize(
    "identity",
    [
        ("production", *IDENTITY[1:]),
        ("notify008", "postgres", *IDENTITY[2:]),
        ("notify008", "notify008", "127.0.0.1", "", "/database/data"),
        ("notify008", "notify008", None, "localhost", "/database/data"),
        ("notify008", "notify008", None, "", "/production/data"),
        None,
    ],
)
def test_socket_user_database_and_storage_path_are_one_identity(identity):
    with pytest.raises(ProbeError, match="fixed Unix-socket"):
        module.database_identity(Connection(identity=identity))


@pytest.mark.parametrize("aurora", [(True,), None, (0,)])
def test_aurora_or_unknown_backend_is_not_an_isolated_postgres_server(aurora):
    with pytest.raises(ProbeError, match="Aurora"):
        module.database_identity(Connection(aurora=aurora))


def test_schema_preparation_is_explicit_once_and_runtime_factory_never_ensures(
    monkeypatch,
):
    connection = Connection()
    calls, closes = [], []
    monkeypatch.setattr(module, "connect", lambda: connection)

    def store(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(close=lambda: closes.append(True))

    monkeypatch.setattr(module, "PostgresStore", store)
    facts = module.prepare_schema(RUN_ID)
    assert facts == {
        "postgres_major": 16,
        "database_is_unix_socket": True,
        "production_credentials_loaded": False,
    }, "preparation must report actual local-backend identity"
    assert calls[0][1]["initialize_schema"] is True and closes == [True], (
        "only the separate preparation step may run DDL and it must close its pool"
    )
    assert any(parameters == (RUN_ID,) for _, parameters in connection.commands), (
        "the private database must persist the run owner through bound parameters"
    )
    runtime = module.PostgresFactory(RUN_ID)()
    runtime.close()
    assert calls[1][1]["initialize_schema"] is False, (
        "the runtime worker must never implicitly initialize or migrate schema"
    )


@pytest.mark.parametrize("tables", [["user_table"], ["notify008_owner"]])
def test_schema_preparation_refuses_existing_data_without_store_construction(
    monkeypatch, tables
):
    connection = Connection(tables=tables)
    monkeypatch.setattr(module, "connect", lambda: connection)
    monkeypatch.setattr(
        module,
        "PostgresStore",
        lambda *args, **kwargs: pytest.fail(
            "nonempty database reached schema constructor"
        ),
    )
    with pytest.raises(ProbeError, match="nonempty"):
        module.prepare_schema(RUN_ID)
    assert not any(query.startswith("CREATE") for query, _ in connection.commands), (
        "an existing database must remain unchanged"
    )


@pytest.mark.parametrize("owner", [[], [("foreign",)], [(RUN_ID,), ("foreign",)]])
def test_runtime_factory_refuses_missing_or_ambiguous_owner(monkeypatch, owner):
    monkeypatch.setattr(module, "connect", lambda: Connection(owner=owner))
    monkeypatch.setattr(
        module,
        "PostgresStore",
        lambda *args, **kwargs: pytest.fail("wrong owner reached runtime"),
    )
    with pytest.raises(ProbeError, match="owner differs"):
        module.PostgresFactory(RUN_ID)()
