from datetime import datetime, timezone
from types import SimpleNamespace

from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store.shared.remote_helpers import remote_command_stats


def _failed_command(error: str) -> SimpleNamespace:
    return SimpleNamespace(
        status=RemoteCommandStatus.FAILED,
        status_source="executor-internal-error",
        error=error,
        cluster_id="gpu-a",
        created_at=datetime.now(timezone.utc),
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


def test_actual_executor_defect_remains_an_internal_error() -> None:
    stats = remote_command_stats([_failed_command("AttributeError: core")])

    assert stats["executor_internal_error_total"] == 1
