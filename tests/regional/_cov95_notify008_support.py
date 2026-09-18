from __future__ import annotations

import multiprocessing
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.e2e.regional.probes import notify008_protocol as contract
from scripts.e2e.regional.probes.notify008_provider import serve_provider

RUN_ID = "notify008-0123456789abcdef"


def variant_record(variant, *, first_sequence=1, first_pid=31, replacement_pid=32):
    receipts = [
        {
            "run_id": RUN_ID,
            "notification_id": variant.notification_id(RUN_ID),
            "message_id": f"simulated-{first_sequence + index:032x}",
            "provider_pid": 22,
            "sequence": first_sequence + index,
        }
        for index in range(variant.total_acceptances)
    ]
    return {
        "variant": variant.key,
        "notification_id": variant.notification_id(RUN_ID),
        "crash_exitcode": -9,
        "before_result": "SENT" if variant.crash == "committed-before-ack" else None,
        "final_result": "SENT",
        "stored_notifications": 1,
        "replay_added_acceptances": 0,
        "early_retry_added_acceptances": 0,
        "exactly_once_proven": False,
        "duplicate_observed": variant.crash == "accepted-before-commit",
        "supervisor_pid": 11,
        "provider_pid": 22,
        "first_pid": first_pid,
        "replacement_pid": replacement_pid,
        "before_receipts": deepcopy(receipts[: variant.first_acceptances]),
        "receipts": receipts,
        "provider_message_id": receipts[-1]["message_id"],
    }


def report_record():
    rows, sequence = [], 1
    for index, variant in enumerate(contract.variants()):
        rows.append(
            variant_record(
                variant,
                first_sequence=sequence,
                first_pid=31 + index * 2,
                replacement_pid=32 + index * 2,
            )
        )
        sequence += variant.total_acceptances
    return {
        "case_id": contract.CASE_ID,
        "run_id": RUN_ID,
        "validation_scope": contract.SCOPE,
        "provider": "SIMULATED",
        "postgres_major": 16,
        "database_is_unix_socket": True,
        "production_credentials_loaded": False,
        "exactly_once_proven": False,
        "children_reaped": True,
        "variants": rows,
    }


@dataclass
class Provider:
    port: int
    pid: int
    ledger: Path


@dataclass(frozen=True)
class SQLiteSimulator:
    path: Path

    def __call__(self):
        from gpu_fault.store import SqliteStore

        return SqliteStore(str(self.path))


@pytest.fixture(name="provider")
def provider_fixture(tmp_path):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    ledger = tmp_path / "provider.jsonl"
    process = context.Process(target=serve_provider, args=(ledger, RUN_ID, sender, 15))
    process.start()
    sender.close()
    try:
        assert receiver.poll(5), "the owned simulated provider did not become ready"
        ready = receiver.recv()
        assert set(ready) == {"pid", "port"}, "the provider failed before readiness"
        assert ready["pid"] == process.pid, (
            "the receipt process must be the spawned child"
        )
        yield Provider(port=ready["port"], pid=ready["pid"], ledger=ledger)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        receiver.close()
        assert not process.is_alive(), "the local provider child was not reaped"
        process.close()
