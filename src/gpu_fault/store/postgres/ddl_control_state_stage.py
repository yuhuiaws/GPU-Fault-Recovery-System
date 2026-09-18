"""The ordered schema stage for optional control-state migrations."""

from __future__ import annotations

from typing import Any

from gpu_fault.store.postgres.ddl_control_state import (
    create_control_state_registry,
    create_control_state_time_functions,
    create_legacy_control_state_fence,
)
from gpu_fault.store.postgres.ddl_control_state_writes import (
    create_control_state_delete_writer,
    create_control_state_lock_reader,
    create_control_state_mirror,
    create_control_state_patch_writer,
    create_control_state_writer,
    create_native_state_writer,
)
from gpu_fault.store.postgres.ddl_remote_command_state import (
    create_control_records_view,
    create_remote_command_payload_functions,
    create_remote_command_state_indexes,
    create_remote_command_state_table,
    create_remote_command_state_view,
    create_remote_command_state_wakeup,
)
from gpu_fault.store.postgres.ddl_workflow_state import (
    create_workflow_payload_functions,
    create_workflow_state_indexes,
    create_workflow_state_table,
    create_workflow_state_view,
    create_workflow_state_wakeup,
)


def create_control_state_stage(cursor: Any) -> None:
    create_control_state_registry(cursor)
    create_control_state_time_functions(cursor)
    create_legacy_control_state_fence(cursor)
    create_remote_command_state_table(cursor)
    create_remote_command_payload_functions(cursor)
    create_workflow_state_table(cursor)
    create_workflow_payload_functions(cursor)
    create_native_state_writer(cursor)
    create_control_state_writer(cursor)
    create_control_state_mirror(cursor)
    create_control_state_patch_writer(cursor)
    create_control_state_lock_reader(cursor)
    create_control_state_delete_writer(cursor)
    create_remote_command_state_indexes(cursor)
    create_remote_command_state_view(cursor)
    create_remote_command_state_wakeup(cursor)
    create_workflow_state_indexes(cursor)
    create_workflow_state_view(cursor)
    create_workflow_state_wakeup(cursor)
    create_control_records_view(cursor, workflow_enabled=True)
