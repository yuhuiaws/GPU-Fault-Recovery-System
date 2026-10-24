from copy import deepcopy
from datetime import datetime, timedelta, timezone

from gpu_fault.flight_progress import group_progress

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def entry(seq, state="started", **extra):
    return {
        "collective_seq_id": seq,
        "state": state,
        "record_id": seq * 10,
        "profiling_name": "nccl:all_reduce",
        **extra,
    }


def temporal_signal(rank, before, after=None, *, members=None, name="dp", offset=0):
    first_at = NOW + timedelta(seconds=offset)
    group = {"pg_name": name, "pg_members": members or [0, 1, 2]}
    last = max(after or before, key=lambda item: item["collective_seq_id"])
    return {
        "rank": rank,
        "pid": 100 + rank,
        "node_id": f"node-{rank}",
        "process_identity": {"verified": True, "start_time_ticks": 1000},
        "flight_recorder": {"last_entries_by_pg": [{**group, **last}]},
        "flight_recorder_series": {
            "version": "fr-series/v1",
            "clock_uncertainty_seconds": 0,
            "status": "PAIRED",
            "process_start_ticks": 1000,
            "interval_seconds": 2,
            "first_observed_at": first_at.isoformat(),
            "second_requested_at": (first_at + timedelta(seconds=2)).isoformat(),
            "second_observed_at": (first_at + timedelta(seconds=3)).isoformat(),
            "first": {
                "status": "dumped",
                "groups": [{**group, "progress": group_progress(before)}],
            },
            "second": {
                "status": "dumped",
                "groups": [{**group, "progress": group_progress(after or before)}],
            },
        },
    }


def add_series(row):
    entries = row["flight_recorder"].get("last_entries_by_pg") or [
        row["flight_recorder"]["last_entry"]
    ]
    groups = [
        {
            **item,
            "progress": group_progress(
                [
                    entry(
                        item["collective_seq_id"],
                        **{
                            key: value
                            for key, value in item.items()
                            if key != "collective_seq_id"
                        },
                    )
                ]
            ),
        }
        for item in entries
    ]
    row["flight_recorder_series"] = {
        "version": "fr-series/v1",
        "clock_uncertainty_seconds": 0,
        "status": "PAIRED",
        "process_start_ticks": 1000,
        "interval_seconds": 2,
        "first_observed_at": NOW.isoformat(),
        "second_requested_at": (NOW + timedelta(seconds=2)).isoformat(),
        "second_observed_at": (NOW + timedelta(seconds=3)).isoformat(),
        "first": {"status": "dumped", "groups": deepcopy(groups)},
        "second": {"status": "dumped", "groups": deepcopy(groups)},
    }
    return row
