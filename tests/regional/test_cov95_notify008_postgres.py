"""Explicit serial target: real PostgreSQL and process loss, SIMULATED provider."""

from __future__ import annotations

import os
from contextlib import closing
from uuid import uuid4

import pytest

from scripts.e2e.regional.probes.notify008_process import exercise
from scripts.e2e.regional.probes.notify008_protocol import Variant, variant_errors
from tests.regional._cov95_notify008_postgres import isolated_database

pytestmark = pytest.mark.skipif(
    not os.environ.get("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="requires the separately granted serial local PostgreSQL slot",
)


@pytest.mark.parametrize("mode", ["inline", "outbox"])
@pytest.mark.parametrize("kind", ["gpu-reset", "workload-restart"])
@pytest.mark.parametrize(
    "crash", ["before-provider", "accepted-before-commit", "committed-before-ack"]
)
@pytest.mark.allows_cluster_binaries("docker")
def test_postgres_runtime_process_loss_and_independent_acceptance(
    tmp_path, mode, kind, crash
):
    # Docker is used only to inspect the explicitly granted container's owner/port.
    run_id = f"notify008-{uuid4().hex[:16]}"
    variant = Variant(mode, crash, kind)
    with isolated_database() as factory:
        result = exercise(factory, tmp_path, run_id, selected=(variant,), seconds=90)
        (row,) = result["variants"]
        assert variant_errors(row, run_id=run_id, variant=variant) == [], (
            "actual PostgreSQL, process loss and independent provider receipts must agree"
        )
        assert row["crash_exitcode"] == -9, (
            "the old runtime must actually terminate at the controlled process boundary"
        )
        assert result["children_reaped"] is True, (
            "all worker/provider children must be reaped"
        )
        assert result["provider"] == "SIMULATED", (
            "this proof does not execute SNS or SES"
        )
        assert row["exactly_once_proven"] is False, (
            "PostgreSQL cannot atomically commit an external provider acceptance"
        )
        assert len(row["receipts"]) == (
            2 if crash == "accepted-before-commit" else 1
        ), "uncommitted acceptance must retain the observable duplicate-delivery window"
        with closing(factory()) as observer:
            saved = observer.get_notification_result(variant.notification_id(run_id))
            assert saved.status.value == "SENT", (
                "a new SQL observer must see the committed result"
            )
            assert saved.provider_message_id == row["receipts"][-1]["message_id"], (
                "the final SQL result must be attributable to the independently observed receipt"
            )
        assert "validation_scope" not in result, (
            "local production-backend proof must not be relabeled as deployed regional evidence"
        )
