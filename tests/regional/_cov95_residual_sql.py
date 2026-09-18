"""Fake psycopg base for exercising the real readonly cursor adapter."""

from __future__ import annotations

import importlib
from collections import deque
from types import SimpleNamespace
from typing import Generic, TypeVar

import pytest

from scripts.e2e.regional.probes import state_table_snapshot
from tests.regional import _cov95_residual_support as support

Row = TypeVar("Row")
residual_isolation = support.residual_isolation


class FakeCursor(Generic[Row]):
    def __init__(self):
        self.commands = []
        self.rows = deque()
        self.plan_row = ([{"Plan": {"Node Type": "Result"}}],)
        self.view_row = ("SELECT id FROM fake_state",)
        self.columns = [SimpleNamespace(name="id", type_code=25)]
        self.description = None

    def execute(self, query, params=None, *, prepare=None, binary=None):
        text = query if isinstance(query, str) else query.as_string()
        self.commands.append((text, params, prepare, binary))
        if text.startswith("EXPLAIN "):
            self.rows = deque([] if self.plan_row is None else [self.plan_row])
        elif text.startswith("SELECT * FROM "):
            self.description = self.columns
            self.rows.clear()
        elif text == "SELECT pg_get_viewdef(to_regclass(%s), true)":
            self.rows = deque([] if self.view_row is None else [self.view_row])
        else:
            self.rows = deque([("ordinary-result",)])
        return self

    def fetchone(self):
        return self.rows.popleft() if self.rows else None


@pytest.fixture
def readonly_cursor(monkeypatch, residual_isolation):
    with monkeypatch.context() as patch:
        patch.setattr("psycopg.Cursor", FakeCursor)
        module = importlib.reload(state_table_snapshot)
        yield module.ReadOnlySchemaCursor()
    importlib.reload(state_table_snapshot)
