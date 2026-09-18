from __future__ import annotations

import os
from contextlib import closing, contextmanager

import pytest

from gpu_fault.store import PostgresStore
from tests.store._postgres_processor_claim_support import postgres_store_instance


def validated_url():
    value = os.environ.get("GPU_FAULT_TEST_POSTGRES_URL")
    if not value:
        pytest.skip(
            "runtime Store tests require the separately allocated PostgreSQL slot"
        )
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("runtime Store PostgreSQL tests require explicit serial -n0")
    from scripts.e2e.regional.run_cap005_postgres_suite import validate_server

    validate_server(value)
    return value


@pytest.fixture(params=["legacy", "dual", "dedicated"])
def pg_store(request, monkeypatch):
    url = validated_url()
    monkeypatch.setenv("GPU_FAULT_POSTGRES_HOT_STATE_MODE", "legacy")
    with closing(postgres_store_instance()) as setup:
        legacy = next(setup)
        if request.param == "legacy":
            yield legacy
            return
        with closing(
            PostgresStore(url, initialize_schema=False, hot_state_mode=request.param)
        ) as store:
            yield store


@contextmanager
def peer_store(mode):
    with closing(
        PostgresStore(validated_url(), initialize_schema=False, hot_state_mode=mode)
    ) as store:
        yield store
