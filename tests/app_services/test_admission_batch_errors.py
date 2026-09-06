"""One store error inside a batch is classified, not blindly re-raised, and
the flush loop cannot die leaving futures pending.

FINAL-建议汇总 F-E3 (P1-74D, P1-74F). ``_run_group`` handed whatever the
store raised to all 64 futures: a retryable serialization failure surfaced as
a 500 to every caller, and a bug in the loop body itself ended the flush task
with the pending futures never answered.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gpu_fault.async_store import AsyncStoreExecutor, StoreIoCapacityExceeded
from tests.store.test_async_store import _admission_batcher


def _submit_one(batcher, item):
    async def scenario():
        return await asyncio.wait_for(batcher.submit(item), timeout=2.0)

    return asyncio.run(scenario())


def test_a_retryable_store_error_becomes_a_capacity_rejection():
    psycopg = pytest.importorskip("psycopg")
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=2, admission_timeout_seconds=2
    )

    def admit_batch(items, **_kwargs):
        raise psycopg.errors.lookup("40001")("could not serialize access")

    batcher = _admission_batcher(
        SimpleNamespace(), executor, flush_delay_seconds=0, admit_batch=admit_batch
    )

    with pytest.raises(StoreIoCapacityExceeded):
        _submit_one(batcher, SimpleNamespace(cluster_id="a", path="/v1/x"))
    executor.close()


def test_a_programming_error_still_surfaces_as_itself():
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=2, admission_timeout_seconds=2
    )

    def admit_batch(items, **_kwargs):
        raise KeyError("missing column")

    batcher = _admission_batcher(
        SimpleNamespace(), executor, flush_delay_seconds=0, admit_batch=admit_batch
    )

    with pytest.raises(KeyError):
        _submit_one(batcher, SimpleNamespace(cluster_id="a", path="/v1/x"))
    executor.close()


def test_a_broken_flush_loop_fails_its_pending_futures_instead_of_hanging(monkeypatch):
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=2, admission_timeout_seconds=2
    )
    batcher = _admission_batcher(SimpleNamespace(), executor, flush_delay_seconds=0)

    def broken(_now):
        raise RuntimeError("dispatch bookkeeping corrupted")

    monkeypatch.setattr(batcher, "_dispatch_locked", broken)

    with pytest.raises(RuntimeError, match="dispatch bookkeeping corrupted"):
        _submit_one(batcher, SimpleNamespace(cluster_id="a", path="/v1/x"))
    executor.close()
