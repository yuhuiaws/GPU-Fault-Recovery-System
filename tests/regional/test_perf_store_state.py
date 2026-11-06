"""A round's store-state label is derived from the store at its start.

Perf plan §3.3: ``optimistic_bound`` (the queue indexes compact: an upper
bound, never the formal record) and ``steady_state`` (the queue indexes at a
working set's high-water mark on the detached layout) are decided by
``store_bloat.before.gpu_fault_processor_queue.index_pages_per_live_row``,
its ``index_min_entries_per_page`` and ``store_layout.mode`` -- "never filled
in by judgement". The runners wrote the two fields but no label, so every
round's label was a reviewer's reading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.perf import regional_capacity_database as database
from scripts.perf import regional_capacity_suite as capacity
from tests.regional._perf_caller_support import CapacityWire, capacity_args

PAGE = database.PAGE_BYTES


def start_of_round(
    *, live_rows: int, index_pages: int, mode: str | None, entries: float = -1
) -> tuple[dict[str, Any], dict[str, Any]]:
    """store_bloat and store_layout as a runner derives them from the
    ``postgres-before.json`` snapshot of a queue with one index."""

    snapshot = {
        "hot_tables": {
            "gpu_fault_processor_queue": {
                "live_rows": live_rows,
                "heap_bytes": 10 * PAGE,
                "index_bytes": index_pages * PAGE,
                "indexes": {
                    "queue_claim": {"bytes": index_pages * PAGE, "entries": entries}
                },
            }
        },
        "hot_layout": {"mode": mode} if mode else None,
    }
    return (
        database.store_bloat_section(snapshot, {}),
        database.store_layout_section(snapshot, {}),
    )


@pytest.mark.parametrize(
    ("round_start", "label", "reason"),
    [
        (
            {"live_rows": 1000, "index_pages": 400, "mode": "detached"},
            "optimistic_bound",
            "compact queue indexes: an upper bound, never the formal record",
        ),
        (
            {"live_rows": 1000, "index_pages": 400, "mode": "inline", "entries": 40000},
            "optimistic_bound",
            "compact queue indexes: an upper bound, never the formal record",
        ),
        (
            {
                "live_rows": 1000,
                "index_pages": 400,
                "mode": "detached",
                "entries": 4000,
            },
            None,
            "queue indexes compact by pages but sparse by entries: no label",
        ),
        (
            {"live_rows": 1000, "index_pages": 19000, "mode": "detached"},
            "steady_state",
            "queue indexes at a working set's high-water mark, detached layout",
        ),
        (
            {"live_rows": 1000, "index_pages": 25000, "mode": "inline"},
            None,
            "layout inline: a comparison at most, not the v23 formal record",
        ),
        (
            {"live_rows": 1000, "index_pages": 4000, "mode": "detached"},
            None,
            "queue indexes between the two labels' bounds: no label",
        ),
    ],
)
def test_the_label_follows_the_queue_indexes_and_the_layout(
    round_start: dict[str, Any], label: str | None, reason: str
) -> None:
    bloat, layout = start_of_round(**round_start)

    state = database.store_state_section(bloat, layout)

    assert state["label"] == label, state
    assert state["reason"] == reason, state
    assert state["queue_index_pages_per_live_row"] == pytest.approx(
        round_start["index_pages"] / round_start["live_rows"]
    ), "the record keeps the value the label was derived from"
    assert state["layout_mode"] == round_start["mode"], state
    if label == "steady_state":
        assert state["requires"] == "no manual store-maintenance before the round", (
            "the one condition the store cannot show is named for the record"
        )
    else:
        assert "requires" not in state, state


def test_a_round_without_the_store_snapshot_has_no_label() -> None:
    bloat = database.store_bloat_section({}, {})
    layout = database.store_layout_section({}, {})

    state = database.store_state_section(bloat, layout)

    assert state["label"] is None, state
    assert state["reason"] == "store_bloat.before was not recorded: no label"


def test_the_burst_summary_carries_the_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    CapacityWire(tmp_path, monkeypatch)

    assert capacity.main(capacity_args("all", tmp_path)) == 0

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["store_state"] == database.store_state_section(
        summary["store_bloat"], summary["store_layout"]
    ), "summary.json.store_state is derived from the two recorded fields"
