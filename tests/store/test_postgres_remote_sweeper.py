"""The unclaimed-command sweeper must not overwrite a lease issued under it.

FINAL-建议汇总 F-D12 (P1-12C). ``expire_unclaimed_remote_commands`` held only
its own global advisory lock, read the PENDING candidates without a row lock
and then wrote whole rows back. A claim that leased one of those candidates
in between -- it takes the per-command lock, which the sweeper never asked
for -- had its lease replaced by FAILED/unclaimed-deadline the moment the
sweeper's upsert got the row.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from threading import Thread

import pytest

from gpu_fault.remote_command_models import RemoteCommandStatus
from tests.store._postgres_processor_claim_support import (
    _command,
    _reload,
    _truncate,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is required",
)


@pytest.fixture
def store():
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def test_postgres_sweeper_does_not_overwrite_a_fresh_lease(store) -> None:
    import psycopg

    now = datetime.now(timezone.utc)
    command = _command(store, "cmd-fresh-lease", request_id="wf-fresh-lease")
    aged = command.model_copy(update={"created_at": now - timedelta(hours=1)})
    store._put("remote_command", command.command_id, aged)

    # An executor's claim, still uncommitted: the per-command lock the claim
    # path takes, and the row rewritten as LEASED.
    leased = aged.model_copy(
        update={
            "status": RemoteCommandStatus.LEASED,
            "lease_owner": "executor-b",
            "lease_token": "token-b",
            "lease_expires_at": now + timedelta(minutes=5),
            "updated_at": now,
        }
    )
    claimer = psycopg.connect(os.environ["GPU_FAULT_TEST_POSTGRES_URL"])
    claimer.autocommit = False
    try:
        with claimer.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"remote_command/{command.command_id}",),
            )
            cursor.execute(
                """
                UPDATE gpu_fault_objects SET payload=%s::jsonb
                WHERE kind='remote_command' AND key=%s
                """,
                (leased.model_dump_json(), command.command_id),
            )
        outcome: dict[str, int] = {}

        def sweep() -> None:
            outcome["expired"] = store.expire_unclaimed_remote_commands(
                older_than=now - timedelta(minutes=30), limit=10
            )

        sweeper = Thread(target=sweep)
        sweeper.start()
        time.sleep(0.5)
        claimer.commit()
        sweeper.join(timeout=10)
    finally:
        claimer.close()

    assert not sweeper.is_alive(), "the sweeper should finish once the claim commits"
    assert outcome["expired"] == 0
    reloaded = _reload(store, command.command_id)
    assert reloaded.status is RemoteCommandStatus.LEASED
    assert reloaded.lease_owner == "executor-b"


def test_postgres_sweeper_still_expires_a_genuinely_unclaimed_command(store) -> None:
    now = datetime.now(timezone.utc)
    command = _command(store, "cmd-stale", request_id="wf-stale")
    store._put(
        "remote_command",
        command.command_id,
        command.model_copy(update={"created_at": now - timedelta(hours=1)}),
    )

    expired = store.expire_unclaimed_remote_commands(
        older_than=now - timedelta(minutes=30), limit=10
    )

    assert expired == 1
    assert _reload(store, command.command_id).status is RemoteCommandStatus.FAILED
