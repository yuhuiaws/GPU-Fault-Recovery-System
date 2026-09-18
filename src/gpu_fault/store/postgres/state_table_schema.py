"""Read-only startup checks for the control-state storage contract."""

from __future__ import annotations

from typing import Any

from gpu_fault.store.postgres.ddl_remote_command_state import (
    create_control_records_view,
    create_remote_command_state_view,
)
from gpu_fault.store.postgres.ddl_workflow_state import create_workflow_state_view
from gpu_fault.store.postgres.state_table_definitions import validate_state_definitions
from gpu_fault.store.postgres.state_table_payload import STATE_LAYOUTS
from gpu_fault.store.postgres.state_table_storage import ENABLED_STATE_KINDS


def validate_state_table_schema(database: Any) -> None:
    with database.cursor() as cursor:
        cursor.execute(
            "SELECT to_regclass('gpu_fault_control_state_modes'), "
            "to_regclass('gpu_fault_control_records'), "
            "to_regprocedure('gpu_fault_put_control_state(text,text,jsonb,jsonb,boolean)'), "
            "to_regprocedure('gpu_fault_patch_control_state(text,text,jsonb)'), "
            "to_regprocedure('gpu_fault_lock_control_state(text,text)'), "
            "to_regprocedure('gpu_fault_delete_control_state(text,text,jsonb,boolean)')"
        )
        if any(value is None for value in cursor.fetchone()):
            raise RuntimeError(
                "control-state schema is incomplete; run --ensure-schema"
            )
        cursor.execute("SELECT kind, mode FROM gpu_fault_control_state_modes")
        modes = dict(cursor.fetchall())
        if set(modes) != set(STATE_LAYOUTS) or any(
            mode not in {"legacy", "dual", "dedicated"} for mode in modes.values()
        ):
            raise RuntimeError("control-state migration modes are invalid")
        for kind, mode in modes.items():
            if kind not in ENABLED_STATE_KINDS and mode != "legacy":
                raise RuntimeError(
                    "a control-state migration is enabled before its schema release"
                )
        triggers = {
            "gpu_fault_objects_control_state_fence_trigger": "gpu_fault_objects",
            "gpu_fault_objects_control_state_mirror_trigger": "gpu_fault_objects",
        }
        for kind in ENABLED_STATE_KINDS:
            layout = STATE_LAYOUTS[kind]
            cursor.execute(
                "SELECT attname, format_type(atttypid, atttypmod) "
                "FROM pg_attribute WHERE attrelid=to_regclass(%s) "
                "AND attnum>0 AND NOT attisdropped",
                (layout.table,),
            )
            actual = dict(cursor.fetchall())
            expected = {
                name: (
                    "timestamp with time zone"
                    if name in layout.datetime_fields
                    else "boolean"
                    if name.endswith("_naive")
                    or name in {"preempt_predecessor", "safety_only"}
                    else "bigint"
                    if name
                    in {
                        "fencing_token",
                        "step_index",
                        "merge_revision",
                        "execution_epoch",
                    }
                    else "jsonb"
                    if name in {layout.payload_column, "result_details"}
                    else "text"
                )
                for name in layout.column_names
            }
            if actual != expected:
                raise RuntimeError(f"{kind} state columns differ from this release")
            triggers.update(
                {
                    f"{layout.table}_fence": layout.table,
                    f"{layout.table}_wakeup_trigger": layout.table,
                }
            )
        cursor.execute(
            "SELECT t.tgname, c.relname, t.tgenabled FROM pg_trigger t "
            "JOIN pg_class c ON c.oid=t.tgrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() "
            "AND t.tgname=ANY(%s) AND NOT t.tgisinternal",
            (sorted(triggers),),
        )
        enabled = {
            name
            for name, table, mode in cursor.fetchall()
            if mode == "O" and table == triggers[name]
        }
        if enabled != set(triggers):
            raise RuntimeError(
                "control-state fencing, mirroring or wakeup triggers are missing or disabled"
            )
        validate_state_definitions(cursor)
    with database.transaction():
        with database.cursor() as cursor:
            create_remote_command_state_view(cursor, validate_only=True)
            if "workflow" in ENABLED_STATE_KINDS:
                create_workflow_state_view(cursor, validate_only=True)
            create_control_records_view(
                cursor,
                workflow_enabled="workflow" in ENABLED_STATE_KINDS,
                validate_only=True,
            )
