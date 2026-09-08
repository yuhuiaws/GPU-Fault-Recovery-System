"""Every writer of a remote command row takes the same lock, checked from source.

Store review 2026-09-07, item A. On Postgres ``_state_transaction(key)`` is an
advisory lock on ``hashtextextended(key)`` and nothing else: the row is read
without a row lock and written back whole by ``_put``. Two writers that spell
the key differently therefore do not exclude each other. That is exactly what
the cancel paths did -- ``remote_command/<id>/timeout`` and
``remote_command/<id>/cancel`` next to the ``remote_command/<id>`` that claim,
completion and lease renewal took -- so a cancel that overlapped a completion
could write the SUCCEEDED command back to LEASED with
``cancellation_requested_at`` set, and the executor's result was lost.

The behavioural half of the fix is ``test_postgres_remote_command_lock_key.py``.
This half stops a future writer from reintroducing a private key: every
``remote_command/...`` literal in the two mixins must be one of the sanctioned
forms, every function that writes a command row must sit under one of the
sanctioned serialisations, and a SQLite writer that relies on the
whole-connection ``BEGIN IMMEDIATE`` instead of a per-command key must be
overridden on Postgres, where that transaction does not exist.
"""

from __future__ import annotations

import ast
import inspect
import re

import pytest

from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres import remote_commands as postgres_module
from gpu_fault.store.postgres.remote_commands import PostgresRemoteCommandMixin
from gpu_fault.store.shared import remote_commands as shared_module
from gpu_fault.store.sqlite import remote_commands as sqlite_module

# The shared module holds the single-row writers both key/value stores run
# (S13); the two dialect modules hold the statements each one adds.
MODULES = {
    "postgres": postgres_module,
    "shared": shared_module,
    "sqlite": sqlite_module,
}

# The per-command key: ``remote_command/{<something>.command_id}`` or
# ``remote_command/{command_id}`` / ``{key}``, and nothing after the id.
PER_COMMAND_KEY = re.compile(r'f"remote_command/\{(?:[\w.]*command_id|key)\}"')
# The same key spelled in SQL for the bulk advisory lock.
SQL_BULK_KEY = "'remote_command/' || command_id"
# The two sweep-level keys; they serialise sweepers against each other and are
# only acceptable together with the row discipline checked below.
SWEEP_KEYS = {
    '"remote_command/cleanup"',
    '"remote_command/unclaimed-expiry"',
    '"remote_command/stale-fence"',
}

LITERAL = re.compile(r"""(f?"remote_command/[^"]*"|'remote_command/'\s*\|\|\s*\w+)""")

WRITE_MARKERS = (
    '_put(\n                "remote_command"',
    '_put(\n                    "remote_command"',
    '_put("remote_command"',
    '_delete("remote_command"',
    "DELETE FROM gpu_fault_objects",
)


def _writer_functions(module) -> dict[str, str]:
    source = inspect.getsource(module)
    tree = ast.parse(source)
    writers: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.get_source_segment(source, node) or ""
        if any(marker in body for marker in WRITE_MARKERS):
            writers[node.name] = body
    return writers


@pytest.mark.parametrize("name", sorted(MODULES))
def test_every_remote_command_lock_literal_is_a_sanctioned_form(name: str) -> None:
    source = inspect.getsource(MODULES[name])
    literals = [match.group(1) for match in LITERAL.finditer(source)]
    assert literals, "the module no longer takes any remote_command lock"
    offending = [
        literal
        for literal in literals
        if not (
            PER_COMMAND_KEY.fullmatch(literal)
            or literal in SWEEP_KEYS
            or " ".join(literal.split()) == SQL_BULK_KEY
        )
    ]
    assert offending == [], (
        f"{name}: private remote_command lock keys reintroduce the lost update "
        f"item A removed; use remote_command/<command_id>: {offending}"
    )


@pytest.mark.parametrize("name", sorted(MODULES))
def test_the_writer_set_is_the_reviewed_one(name: str) -> None:
    expected = {
        "postgres": {
            "claim_remote_commands",
            "cancel_remote_command",
            "cancel_remote_commands_for_workflow",
            "cleanup_terminal_remote_commands",
            "expire_unclaimed_remote_commands",
            "expire_stale_fenced_remote_commands",
        },
        "shared": {
            "ensure_remote_command",
            "renew_remote_command_lease",
            "complete_remote_command",
        },
        "sqlite": {
            "claim_remote_commands",
            "expire_unclaimed_remote_commands",
            "expire_stale_fenced_remote_commands",
            "cleanup_terminal_remote_commands",
            "cancel_remote_commands_for_workflow",
            "cancel_remote_command",
        },
    }[name]
    assert set(_writer_functions(MODULES[name])) == expected, (
        f"{name}: a remote_command writer was added or removed; review its lock "
        "key against the docstring of this test and update the set"
    )


def _serialisation(body: str) -> str:
    """Name the serialisation a writer body uses, or ``"none"``."""

    if PER_COMMAND_KEY.search(body):
        return "per-command"
    if (
        "pg_advisory_xact_lock" in body
        and SQL_BULK_KEY in " ".join(body.split())
        and "ORDER BY command_id" in body
    ):
        return "bulk-ordered"
    if '"remote_command/cleanup"' in body and "FOR UPDATE SKIP LOCKED" in body:
        if "'SUCCEEDED', 'FAILED'" in body:
            return "terminal-cleanup"
    if '"remote_command/cleanup"' in body and "RemoteCommandStatus.SUCCEEDED" in body:
        return "terminal-cleanup"
    if any(key in body for key in SWEEP_KEYS - {'"remote_command/cleanup"'}):
        return "sweep-key-only"
    return "none"


@pytest.mark.parametrize("name", ["postgres", "shared"])
def test_every_writer_postgres_runs_serialises_on_the_per_command_lock(
    name: str,
) -> None:
    """Postgres has no whole-connection transaction to fall back on, and the
    shared writers run on Postgres unchanged."""

    for function, body in sorted(_writer_functions(MODULES[name]).items()):
        assert _serialisation(body) in {
            "per-command",
            "bulk-ordered",
            "terminal-cleanup",
        }, (
            f"{name} {function}: writes a remote_command row without the "
            "remote_command/<command_id> advisory lock (or the ordered bulk "
            "form / terminal cleanup)"
        )


def test_sqlite_writers_without_a_per_command_key_are_overridden_on_postgres() -> None:
    """The SQLite sweeps rely on ``BEGIN IMMEDIATE``; Postgres must not run them.

    ``PostgresStore`` no longer inherits from ``SqliteStore`` (S13), so a
    SQLite-only writer cannot reach Postgres by accident; this pins that every
    such writer has a Postgres statement of its own rather than a shared one.
    """

    for function, body in sorted(_writer_functions(sqlite_module).items()):
        serialisation = _serialisation(body)
        if serialisation == "per-command":
            continue
        assert serialisation in {"terminal-cleanup", "sweep-key-only"}, (
            f"sqlite {function}: unknown serialisation {serialisation}"
        )
        assert getattr(PostgresStore, function) is getattr(
            PostgresRemoteCommandMixin, function
        ), (
            f"sqlite {function} takes no per-command key and is not overridden "
            "on Postgres, where the advisory lock is the only exclusion"
        )


def test_postgres_cancel_is_not_the_inherited_read_without_row_lock() -> None:
    """Item A: the single-command cancel reads ``FOR UPDATE`` like the bulk one."""

    assert (
        PostgresStore.cancel_remote_command
        is PostgresRemoteCommandMixin.cancel_remote_command
    )
    body = inspect.getsource(PostgresRemoteCommandMixin.cancel_remote_command)
    assert '_get_for_update("remote_command"' in body
    assert "_get_optional(" not in body
