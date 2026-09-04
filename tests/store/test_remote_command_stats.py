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
