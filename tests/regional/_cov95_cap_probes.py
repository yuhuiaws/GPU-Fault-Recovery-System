from __future__ import annotations

import subprocess
from typing import Any


class RecordingStore:
    def __init__(self, events: list[str], *, fail_after: int | None = None) -> None:
        self.events = events
        self.commands: list[Any] = []
        self.fail_after = fail_after
        self.closed = False

    def ensure_remote_command(self, command: Any) -> None:
        if len(self.commands) == self.fail_after:
            raise RuntimeError("fake command insertion failed")
        self.commands.append(command)

    def close(self) -> None:
        self.closed = True
        self.events.append("store-close")


class RecordingDatabase:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.connections: list[tuple[str, dict[str, Any]]] = []
        self.statements: list[tuple[str, Any]] = []
        self.closed = 0
        self.fail_statement: str | None = None

    def connect(self, url: str, **kwargs: Any) -> RecordingConnection:
        self.connections.append((url, kwargs))
        return RecordingConnection(self)


class RecordingConnection:
    def __init__(self, database: RecordingDatabase) -> None:
        self.database = database

    def __enter__(self) -> RecordingConnection:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.database.closed += 1

    def cursor(self) -> RecordingCursor:
        return RecordingCursor(self.database)


class RecordingCursor:
    def __init__(self, database: RecordingDatabase) -> None:
        self.database = database

    def __enter__(self) -> RecordingCursor:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def execute(self, statement: Any, parameters: Any = None) -> None:
        from psycopg import sql

        text = (
            statement.as_string()
            if isinstance(statement, sql.Composable)
            else statement
        )
        self.database.statements.append((text, parameters))
        if text.startswith("DROP DATABASE"):
            self.database.events.append("drop")
        elif text.startswith("CREATE DATABASE"):
            self.database.events.append("create")
        if self.database.fail_statement and text.startswith(
            self.database.fail_statement
        ):
            raise RuntimeError("fake database operation failed")


class RecordingProcess:
    pid = 12345

    def __init__(
        self, events: list[str], *, exit_code: int | None, wait_timeout: bool = False
    ) -> None:
        self.events = events
        self.exit_code = exit_code
        self.wait_timeout = wait_timeout
        self.polls = 0

    def poll(self) -> int | None:
        self.polls += 1
        return self.exit_code

    def terminate(self) -> None:
        self.events.append("terminate")

    def kill(self) -> None:
        self.events.append("kill")
        self.exit_code = -9

    def wait(self, *, timeout: int) -> int:
        self.events.append(f"wait:{timeout}")
        if self.wait_timeout:
            self.wait_timeout = False
            raise subprocess.TimeoutExpired("fake-cap-child", timeout)
        self.exit_code = self.exit_code or 0
        return self.exit_code
