"""Typed control-state I/O behind the existing Store primitives."""

from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any

from pydantic import BaseModel

from gpu_fault.models import datetime_json_text
from gpu_fault.store.postgres.state_table_payload import (
    STATE_LAYOUTS,
    split_state_record,
    state_update_columns,
)
from gpu_fault.store.shared.errors import NotFoundError, StaleWriteError

ENABLED_STATE_KINDS = frozenset(STATE_LAYOUTS)


def put_state_record(
    database: Any,
    kind: str,
    key: str,
    value: BaseModel,
    *,
    expected: BaseModel | None = None,
    guard_versions: bool = False,
) -> bool:
    layout = STATE_LAYOUTS[kind]
    if expected is not None and any(
        getattr(item, layout.key_field, None) != key for item in (value, expected)
    ):
        raise StaleWriteError(
            f"{kind}/{key} differs from the conditional record identity"
        )
    columns = split_state_record(layout, value)
    with (
        database.transaction() if expected is not None else nullcontext(),
        database.cursor() as cursor,
    ):
        expected_json = None
        if expected is not None:
            # Models normalize old defaults and timestamps. Lock the authoritative
            # row, compare that model, then retain its exact JSON for the SQL CAS.
            cursor.execute(
                "SELECT payload FROM gpu_fault_lock_control_state(%s, %s)",
                (kind, key),
            )
            row = cursor.fetchone()
            if row is None:
                raise StaleWriteError(f"{kind}/{key} changed since it was read")
            current = type(expected).model_validate(row[0])
            if current.model_dump(mode="json") != expected.model_dump(mode="json"):
                raise StaleWriteError(f"{kind}/{key} changed since it was read")
            expected_json = json.dumps(row[0])
        cursor.execute(
            "SELECT gpu_fault_put_control_state(%s, %s, %s::jsonb, %s::jsonb, %s)",
            (
                kind,
                key,
                json.dumps(columns, default=datetime_json_text),
                expected_json,
                guard_versions,
            ),
        )
        applied = cursor.fetchone()[0]
        if not isinstance(applied, bool):
            raise RuntimeError("control-state writer returned an invalid result")
        if not applied and expected is not None:
            raise StaleWriteError(f"{kind}/{key} changed since it was read")
    return applied


def put_state_fields(
    database: Any,
    kind: str,
    key: str,
    value: BaseModel,
    fields: frozenset[str],
) -> None:
    columns = state_update_columns(STATE_LAYOUTS[kind], value, fields)
    with database.cursor() as cursor:
        cursor.execute(
            "SELECT gpu_fault_patch_control_state(%s, %s, %s::jsonb)",
            (kind, key, json.dumps(columns, default=datetime_json_text)),
        )
        if not cursor.fetchone()[0]:
            raise StaleWriteError(f"{kind}/{key} changed since it was read")


def get_state_payload(
    database: Any,
    kind: str,
    key: str,
    *,
    for_update: bool = False,
) -> Any:
    query = (
        "SELECT payload FROM gpu_fault_lock_control_state(%s, %s)"
        if for_update
        else "SELECT payload FROM gpu_fault_control_records WHERE kind=%s AND key=%s"
    )
    with database.cursor() as cursor:
        cursor.execute(query, (kind, key))
        row = cursor.fetchone()
    if row is None:
        raise NotFoundError(key)
    return row[0]


def list_state_payloads(database: Any, kind: str) -> list[Any]:
    with database.cursor() as cursor:
        cursor.execute(
            "SELECT payload FROM gpu_fault_control_records WHERE kind=%s", (kind,)
        )
        return [row[0] for row in cursor.fetchall()]
