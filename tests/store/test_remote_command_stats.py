from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store.postgres.remote_commands import PostgresRemoteCommandMixin
from gpu_fault.store.shared.remote_helpers import remote_command_stats

NOW = datetime(2026, 9, 2, 14, 0, tzinfo=timezone.utc)


def _failed_command(error: str, *, updated_at: datetime = NOW) -> SimpleNamespace:
    return SimpleNamespace(
        status=RemoteCommandStatus.FAILED,
        status_source="executor-internal-error",
        error=error,
        cluster_id="gpu-a",
        created_at=updated_at,
        updated_at=updated_at,
        lease_owner=None,
    )


def test_legacy_isolation_conflict_is_not_counted_as_internal_error() -> None:
    stats = remote_command_stats(
        [
            _failed_command(
                "ValueError: node is already isolated by another incident/token"
            )
        ]
    )

    assert stats["by_status"]["FAILED"] == 1
    assert stats["executor_internal_error_total"] == 0
    assert stats["executor_internal_error_last_seen_timestamp_seconds"] == 0.0


def test_actual_executor_defect_remains_an_internal_error() -> None:
    stats = remote_command_stats([_failed_command("AttributeError: core")])

    assert stats["executor_internal_error_total"] == 1
    assert (
        stats["executor_internal_error_last_seen_timestamp_seconds"] == NOW.timestamp()
    )


def test_internal_error_timestamp_tracks_newest_retained_defect() -> None:
    latest = NOW + timedelta(seconds=30)

    stats = remote_command_stats(
        [
            _failed_command("AttributeError: first"),
            _failed_command("RuntimeError: second", updated_at=latest),
        ]
    )

    assert stats["executor_internal_error_total"] == 2
    assert (
        stats["executor_internal_error_last_seen_timestamp_seconds"]
        == latest.timestamp()
    )


def test_postgres_internal_error_stats_include_latest_timestamp() -> None:
    class Cursor:
        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, query: str, parameters: tuple[object, ...]) -> None:
            assert "max(" in query
            assert len(parameters) == 3

        def fetchall(self) -> list[tuple[object, ...]]:
            return [("gpu-a", RemoteCommandStatus.FAILED.value, 2, 1, NOW, None, 0)]

    store = PostgresRemoteCommandMixin()
    setattr(store, "_db", SimpleNamespace(cursor=lambda: Cursor()))

    stats = store.remote_command_stats(now=NOW)

    assert stats["total"] == 2
    assert stats["executor_internal_error_total"] == 1
    assert (
        stats["executor_internal_error_last_seen_timestamp_seconds"] == NOW.timestamp()
    )


def test_remote_command_by_status_counts_per_cluster() -> None:
    """The SQL groups by (cluster, status); the totals must add the groups up.

    FINAL-建议汇总 F-D12 (P1-12D): ``by_status`` took the *last* cluster's
    count for each status, so a fleet with three GPU clusters reported one
    cluster's backlog as the whole fleet's.
    """

    class Cursor:
        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, query: str, parameters: tuple[object, ...]) -> None:
            assert "GROUP BY" in query, "the stats query must aggregate in SQL"

        def fetchall(self) -> list[tuple[object, ...]]:
            leased = RemoteCommandStatus.LEASED.value
            pending = RemoteCommandStatus.PENDING.value
            return [
                ("gpu-a", leased, 2, 0, None, None, 0),
                ("gpu-b", leased, 3, 0, None, None, 0),
                ("gpu-b", pending, 1, 0, None, NOW - timedelta(seconds=40), 0),
                ("gpu-c", leased, 4, 0, None, None, 0),
            ]

    store = PostgresRemoteCommandMixin()
    setattr(store, "_db", SimpleNamespace(cursor=lambda: Cursor()))

    stats = store.remote_command_stats(now=NOW)

    assert stats["total"] == 10
    assert stats["by_status"]["LEASED"] == 9
    assert stats["by_status"]["PENDING"] == 1
    assert stats["open_by_cluster"] == {"gpu-a": 2, "gpu-b": 4, "gpu-c": 4}
    assert stats["oldest_unclaimed_age_seconds_by_cluster"] == {"gpu-b": 40.0}


def _open_command(cluster_id: str, status: RemoteCommandStatus) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        status_source=None,
        error=None,
        cluster_id=cluster_id,
        created_at=NOW - timedelta(seconds=40),
        updated_at=NOW - timedelta(seconds=40),
        lease_owner="executor-a" if status is RemoteCommandStatus.LEASED else None,
    )


def test_stats_break_status_counts_down_by_cluster_in_both_implementations() -> None:
    """F-D12: ``by_status`` is fleet-wide; the per-cluster gauge needs the
    (cluster, status) cells, so both implementations publish the same
    ``by_cluster_status`` shape for the same commands."""

    leased = RemoteCommandStatus.LEASED
    pending = RemoteCommandStatus.PENDING
    failed = RemoteCommandStatus.FAILED
    commands = [
        _open_command("gpu-a", leased),
        _open_command("gpu-a", leased),
        _open_command("gpu-b", leased),
        _open_command("gpu-b", pending),
        _failed_command("AttributeError: core"),
    ]

    class Cursor:
        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, query: str, parameters: tuple[object, ...]) -> None:
            assert "GROUP BY" in query, "the stats query must aggregate in SQL"

        def fetchall(self) -> list[tuple[object, ...]]:
            return [
                ("gpu-a", leased.value, 2, 0, None, None, 0),
                ("gpu-a", failed.value, 1, 1, NOW, None, 0),
                ("gpu-b", leased.value, 1, 0, None, None, 0),
                ("gpu-b", pending.value, 1, 0, None, NOW - timedelta(seconds=40), 0),
            ]

    postgres = PostgresRemoteCommandMixin()
    setattr(postgres, "_db", SimpleNamespace(cursor=lambda: Cursor()))

    shared = remote_command_stats(commands, now=NOW)
    from_sql = postgres.remote_command_stats(now=NOW)

    expected = {
        "gpu-a": {"LEASED": 2, "FAILED": 1},
        "gpu-b": {"LEASED": 1, "PENDING": 1},
    }
    assert shared["by_cluster_status"] == expected
    assert from_sql["by_cluster_status"] == expected
    assert shared["by_status"] == from_sql["by_status"]
    assert shared["open_by_cluster"] == from_sql["open_by_cluster"]
    assert shared["total"] == from_sql["total"] == 5


def test_a_fleet_without_commands_has_an_empty_cluster_breakdown() -> None:
    assert remote_command_stats([], now=NOW)["by_cluster_status"] == {}
