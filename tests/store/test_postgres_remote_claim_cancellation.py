"""A remote command whose cancellation was requested is never claimed again.

FINAL-建议汇总 F-D12 (P0-12B). The memory and sqlite claims filter
``cancellation_requested_at``; the Postgres claim did not -- neither in its
SQL nor in the Python pass after it -- so a LEASED command whose owner was
told to stop came back to the next executor as soon as its lease expired.
"""

from __future__ import annotations

import os
import time

import pytest

from gpu_fault.remote_command_models import RemoteCommandStatus
from tests.store._postgres_processor_claim_support import (  # noqa: F401
    _command,
    _reload,
    store,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is not configured",
)


def test_a_cancellation_requested_command_is_not_reclaimed_after_its_lease(store):  # noqa: F811
    command = _command(store, "remote-cancel-me", request_id="workflow-cancel")
    first = store.claim_remote_commands(
        "cluster-a", "executor-a", limit=1, lease_seconds=1
    )
    assert [item.command_id for item in first] == [command.command_id]
    result = store.cancel_remote_commands_for_workflow(
        command.workflow_request_id, reason="workflow superseded"
    )
    assert result == {"cancelled": 0, "cancellation_requested": 1}
    time.sleep(1.2)  # the lease is now expired

    second = store.claim_remote_commands(
        "cluster-a", "executor-b", limit=5, lease_seconds=60
    )

    assert second == []
    current = _reload(store, command.command_id)
    assert current.status is RemoteCommandStatus.LEASED
    assert current.lease_owner == "executor-a"
