"""Terminal quarantine arbitration through each native workflow storage mode."""

from __future__ import annotations

import os
from contextlib import closing

import pytest

from gpu_fault.state_table_migrate import backfill_state_table, set_state_table_mode
from tests._builders import build_context
from tests.orchestration.test_terminal_quarantine_constraints import (
    prove_quarantine_after_completed_stop,
    prove_quarantine_over_pending_reboot,
    prove_reboot_preserves_quarantine,
)

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("GPU_FAULT_TEST_POSTGRES_URL"),
        reason="requires an owned, isolated local PostgreSQL allocation",
    ),
    pytest.mark.allows_cluster_binaries("docker"),
]


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
@pytest.mark.parametrize(
    "scenario",
    [
        prove_quarantine_after_completed_stop,
        prove_quarantine_over_pending_reboot,
        prove_reboot_preserves_quarantine,
    ],
    ids=["post-stop", "temporary-to-persistent", "preserve-persistent"],
)
def test_terminal_quarantine_arbitration_uses_native_store(mode, scenario):
    import psycopg

    from tests.regional._cov95_notify008_postgres import isolated_database

    with isolated_database() as factory:
        if mode != "legacy":
            with psycopg.connect(factory.url, autocommit=True) as connection:
                set_state_table_mode(
                    connection, "workflow", "dual", expected_mode="legacy"
                )
                if mode == "dedicated":
                    backfill_state_table(connection, "workflow")
                    set_state_table_mode(
                        connection,
                        "workflow",
                        "dedicated",
                        expected_mode="dual",
                        confirm_dedicated=True,
                    )
        with closing(factory()) as store:
            scenario(build_context(store=store))
