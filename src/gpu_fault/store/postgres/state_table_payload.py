"""Lossless projections between control-record models and typed state columns.

Lease updates must not serialize or rewrite the embedded command snapshots.
The database timestamps are UTC; a separate flag retains the model's supported
offset-less datetime representation for JSON equality and existing cursors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from gpu_fault.models import datetime_json_text


@dataclass(frozen=True)
class StateTableLayout:
    kind: str
    table: str
    key_field: str
    fields: tuple[str, ...]
    datetime_fields: frozenset[str]
    payload_column: str = "payload"

    @property
    def column_names(self) -> tuple[str, ...]:
        return (
            *self.fields,
            *(f"{name}_naive" for name in self.fields if name in self.datetime_fields),
            *(("execution_owner",) if self.kind == "remote_command" else ()),
            self.payload_column,
        )


REMOTE_COMMAND_LAYOUT = StateTableLayout(
    kind="remote_command",
    table="gpu_fault_remote_commands",
    key_field="command_id",
    payload_column="snapshot",
    fields=(
        "command_id",
        "cluster_id",
        "workflow_request_id",
        "incident_id",
        "step_index",
        "fencing_token",
        "status",
        "lease_owner",
        "last_lease_owner",
        "lease_token",
        "lease_expires_at",
        "cancellation_requested_at",
        "cancellation_reason",
        "result_details",
        "error",
        "status_source",
        "created_at",
        "updated_at",
    ),
    datetime_fields=frozenset(
        {
            "lease_expires_at",
            "cancellation_requested_at",
            "created_at",
            "updated_at",
        }
    ),
)

WORKFLOW_LAYOUT = StateTableLayout(
    kind="workflow",
    table="gpu_fault_workflows",
    key_field="request_id",
    fields=(
        "request_id",
        "incident_id",
        "status",
        "blocked_kind",
        "fencing_token",
        "execution_epoch",
        "execution_owner_id",
        "execution_lease_expires_at",
        "merge_revision",
        "predecessor_workflow_id",
        "preempt_predecessor",
        "preemption_pending_by_workflow_id",
        "runtime_profile_version",
        "safety_only",
        "not_before",
        "failure_handled_at",
        "created_at",
        "updated_at",
    ),
    datetime_fields=frozenset(
        {
            "execution_lease_expires_at",
            "not_before",
            "failure_handled_at",
            "created_at",
            "updated_at",
        }
    ),
)

STATE_LAYOUTS = {
    layout.kind: layout for layout in (REMOTE_COMMAND_LAYOUT, WORKFLOW_LAYOUT)
}


def _state_columns(
    layout: StateTableLayout,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    columns = dict(fields)
    for name in layout.datetime_fields.intersection(fields):
        text = fields[name]
        parsed = datetime.fromisoformat(text) if text is not None else None
        columns[f"{name}_naive"] = parsed is not None and parsed.tzinfo is None
        columns[name] = (
            parsed.replace(tzinfo=UTC)
            if parsed is not None and parsed.tzinfo is None
            else parsed
        )
    return columns


def split_state_record(
    layout: StateTableLayout,
    value: BaseModel,
) -> dict[str, Any]:
    payload = value.model_dump(mode="json")
    missing = set(layout.fields).difference(payload)
    if missing:
        raise ValueError(f"{layout.kind} state fields are missing: {sorted(missing)}")
    columns = _state_columns(
        layout,
        {name: payload.pop(name) for name in layout.fields},
    )
    if layout.kind == "remote_command":
        columns["execution_owner"] = payload["step"]["execution_owner"]
    columns[layout.payload_column] = payload
    return columns


def join_state_record(
    layout: StateTableLayout,
    columns: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(columns[layout.payload_column])
    if set(layout.fields).intersection(payload):
        raise ValueError("state snapshot contains a separately stored field")
    for name in layout.fields:
        value = columns[name]
        if name in layout.datetime_fields:
            naive = columns[f"{name}_naive"]
            if (
                type(naive) is not bool
                or value is not None
                and not isinstance(value, datetime)
            ):
                raise ValueError("invalid state timestamp columns")
            if value is not None:
                if value.tzinfo is None:
                    raise ValueError("database state timestamp has no UTC offset")
                value = value.astimezone(UTC)
                value = datetime_json_text(
                    value.replace(tzinfo=None) if naive else value
                )
            elif naive:
                raise ValueError("a null state timestamp cannot be naive")
        payload[name] = value
    return payload


def state_update_columns(
    layout: StateTableLayout,
    value: BaseModel,
    fields: frozenset[str],
) -> dict[str, Any]:
    if not fields or not fields.issubset(set(layout.fields) - {layout.key_field}):
        raise ValueError("partial state update names invalid columns")
    # include is applied by the model serializer before it traverses snapshots.
    return _state_columns(layout, value.model_dump(mode="json", include=set(fields)))
