"""The v19 storage-parameter stage against cursor doubles (no PostgreSQL)."""

from __future__ import annotations

import pytest

from gpu_fault.store.postgres.ddl_storage_parameters import (
    HOT_TABLE_STORAGE_PARAMETERS,
    create_storage_parameters_stage,
    set_storage_parameters_if_different,
)


class CatalogDouble:
    """Answers the reloptions read with a fixed row; records every statement."""

    def __init__(self, row) -> None:
        self.row = row
        self.statements: list[str] = []

    def execute(self, sql, parameters=None) -> None:
        self.statements.append(" ".join(str(sql).split()))

    def fetchone(self):
        return self.row


def _alters(cursor: CatalogDouble) -> list[str]:
    return [item for item in cursor.statements if item.startswith("ALTER TABLE")]


def test_a_one_column_catalog_row_means_no_toast_options() -> None:
    """DDL test doubles answer every catalog read with a one-column row (the
    release gate's ``RecordingCursor`` returns ``(True,)``); the stage must
    read it as "nothing stored" and never index a second column."""

    cursor = CatalogDouble((True,))
    create_storage_parameters_stage(cursor)

    alters = _alters(cursor)
    assert len(alters) == len(HOT_TABLE_STORAGE_PARAMETERS), alters
    assert (
        "ALTER TABLE gpu_fault_processor_queue SET (autovacuum_analyze_scale_factor=0, "
        "autovacuum_analyze_threshold=50, autovacuum_vacuum_cost_delay=0, "
        "autovacuum_vacuum_scale_factor=0, autovacuum_vacuum_threshold=50, "
        "fillfactor=70, toast.autovacuum_vacuum_cost_delay=0, "
        "toast.autovacuum_vacuum_scale_factor=0, toast.autovacuum_vacuum_threshold=50)"
    ) in alters


@pytest.mark.parametrize("row", [None, (), (None,), (None, None), [None, None]])
def test_absent_or_empty_catalog_rows_never_raise(row) -> None:
    cursor = CatalogDouble(row)
    set_storage_parameters_if_different(
        cursor,
        "gpu_fault_processor_lanes",
        HOT_TABLE_STORAGE_PARAMETERS["gpu_fault_processor_lanes"],
    )
    expected = 0 if row is None else 1
    assert len(_alters(cursor)) == expected, cursor.statements


def test_matching_main_and_toast_options_issue_no_alter() -> None:
    options = HOT_TABLE_STORAGE_PARAMETERS["gpu_fault_processor_queue"]
    main = [f"{k}={v}" for k, v in options.items() if not k.startswith("toast.")]
    toast = [
        f"{k.removeprefix('toast.')}={v}"
        for k, v in options.items()
        if k.startswith("toast.")
    ]
    cursor = CatalogDouble((main, toast))
    set_storage_parameters_if_different(cursor, "gpu_fault_processor_queue", options)
    assert _alters(cursor) == []


def test_only_the_differing_options_are_altered() -> None:
    options = HOT_TABLE_STORAGE_PARAMETERS["gpu_fault_processor_queue"]
    main = [f"{k}={v}" for k, v in options.items() if not k.startswith("toast.")]
    main = [item for item in main if not item.startswith("fillfactor=")]
    toast = ["autovacuum_vacuum_scale_factor=0", "autovacuum_vacuum_cost_delay=0"]
    cursor = CatalogDouble((main, toast))
    set_storage_parameters_if_different(cursor, "gpu_fault_processor_queue", options)
    assert _alters(cursor) == [
        "ALTER TABLE gpu_fault_processor_queue SET "
        "(fillfactor=70, toast.autovacuum_vacuum_threshold=50)"
    ]
