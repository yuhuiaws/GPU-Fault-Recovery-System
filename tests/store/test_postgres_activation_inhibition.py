"""Native claim contracts; run only with the separately allocated serial PG slot."""

from __future__ import annotations

import pytest

from gpu_fault.regional import RemoteCommandStatus
from tests.store import _cov95_runtime_postgres as postgres
from tests.store import test_activation_inhibition_claims as shared

pg_store = postgres.pg_store


@pytest.mark.parametrize("protocol", [1, 2, 3])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("marker", [True, False, None, 0, "true", {}])
def test_native_sql_filters_inhibited_commands_before_limit(
    pg_store, protocol, batched, marker
):
    shared.test_old_protocol_cannot_claim_marker_but_does_not_starve_ordinary_work(
        pg_store, protocol, batched, marker
    )


@pytest.mark.parametrize("batched", [False, True])
def test_native_default_preserves_unmarked_commands(pg_store, batched):
    shared.test_legacy_default_is_safe_without_changing_ordinary_batch_acceptance(
        pg_store, batched
    )


@pytest.mark.parametrize(
    "status", [RemoteCommandStatus.WAITING, RemoteCommandStatus.LEASED]
)
def test_native_expired_and_waiting_commands_keep_protocol_requirement(
    pg_store, status
):
    shared.test_old_executor_cannot_reclaim_waiting_or_expired_inhibited_work(
        pg_store, status
    )


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "before,after", [(True, shared.ABSENT), (shared.ABSENT, True), (True, False)]
)
def test_native_inhibition_authority_is_immutable(pg_store, batched, before, after):
    shared.test_command_identity_cannot_gain_or_lose_inhibition_on_replay(
        pg_store, batched, before, after
    )
