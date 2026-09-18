from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from gpu_fault.store import InMemoryStore
from scripts.e2e.regional.probes import cap004_commands as probe

RUN = "capinventorytest"


@pytest.mark.parametrize("run_id", ["", "../capbad", "capSHORT", "cap" + "x" * 33])
def test_command_inventory_requires_an_owned_run_identity(run_id) -> None:
    with pytest.raises(ValueError, match="run identity"):
        probe.commands_for_run(run_id)


@pytest.mark.parametrize(
    "failure", ["mode", "agent", "duplicate", "foreign", "missing"]
)
def test_inventory_guards_refuse_unknown_or_foreign_records(
    failure, monkeypatch
) -> None:
    store = InMemoryStore()
    row = probe.commands_for_run(RUN)[0]
    if failure == "agent":
        monkeypatch.setattr(store, "list_agents", lambda: [object()])
    elif failure == "duplicate":
        monkeypatch.setattr(store, "list_remote_commands", lambda: [row, row])
    elif failure == "foreign":
        monkeypatch.setattr(
            store,
            "list_remote_commands",
            lambda: [row.model_copy(update={"command_id": "foreign"})],
        )
    with pytest.raises(
        ValueError, match="mode is invalid|Node Agent|unexpected commands|incomplete"
    ):
        probe.command_snapshot(
            store, RUN, "invalid" if failure == "mode" else "inspect"
        )


def test_cleanup_refuses_a_nonterminal_command_without_a_claim_receipt(
    monkeypatch,
) -> None:
    store = InMemoryStore()
    probe.command_snapshot(store, RUN, "seed")
    monkeypatch.setattr(store, "claim_remote_commands", lambda *_a, **_k: [])
    with pytest.raises(ValueError, match="cannot establish command ownership"):
        probe.command_snapshot(store, RUN, "cleanup")
    assert len(store.list_remote_commands()) == 25, (
        "failed cleanup must retain all command evidence"
    )
    assert {item.status.value for item in store.list_remote_commands()} == {
        "PENDING"
    }, "cleanup may not fabricate terminal states without a lease"


@pytest.mark.parametrize("failure", ["", "argument", "file", "connection", "snapshot"])
def test_public_probe_binds_the_database_and_closes_its_fake_store(
    failure, tmp_path, monkeypatch, capsys
) -> None:
    import psycopg

    database = f"gpu_fault_{RUN}_cap004"
    (tmp_path / "database-name").write_text("other" if failure == "file" else database)
    (tmp_path / "store-url").write_text("postgresql://fixture.invalid/disposable")
    connections = []
    stores = []
    closed = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, statement):
            assert statement == "SELECT current_database()", (
                "read identity before opening a Store"
            )

        def fetchone(self):
            return ("other" if failure == "connection" else database,)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            connections.append("closed")

        def cursor(self):
            return Cursor()

    def connect(url, **kwargs):
        connections.append((url, kwargs))
        return Connection()

    def open_store(url, **kwargs):
        stores.append((url, kwargs))
        value = InMemoryStore()
        if failure == "snapshot":
            monkeypatch.setattr(value, "list_agents", lambda: [object()])
        value.close = lambda: closed.append(True)
        return value

    monkeypatch.setattr(probe, "Path", lambda value: tmp_path / Path(value).name)
    monkeypatch.setattr(psycopg, "connect", connect)
    monkeypatch.setattr(probe, "PostgresStore", open_store)
    monkeypatch.setattr(
        sys,
        "argv",
        ["cap004-probe", "seed", RUN, "unowned" if failure == "argument" else database],
    )
    if failure:
        with pytest.raises(ValueError, match="identity|different database|Node Agent"):
            probe.main()
    else:
        assert probe.main() == 0, "the isolated fake database probe should complete"
        result = json.loads(capsys.readouterr().out)
        assert result["commands_created"] == 25, (
            "seed exactly the documented command count"
        )
        assert result["status_counts"] == {"PENDING": 25}, "seeding is not execution"
    if failure in {"argument", "file", "connection"}:
        assert stores == [], (
            "unproved database identity must stop before Store creation"
        )
    else:
        assert closed == [True], (
            "success and snapshot failure must both close the Store"
        )
        assert stores[0][1] == {
            "initialize_schema": False,
            "pool_min_size": 0,
            "pool_max_size": 2,
        }, "the probe must not run DDL or create an unbounded pool"
    if failure not in {"argument", "file"}:
        assert connections[-1] == "closed", "identity-read connection must always close"
