from __future__ import annotations

import json
import sys

from gpu_fault import store_migrate


def test_store_migrate_can_initialize_schema(monkeypatch, capsys) -> None:
    calls = []

    class FakeStore:
        def __init__(self, url, **kwargs) -> None:
            calls.append(("open", url, kwargs))

        def close(self) -> None:
            calls.append(("close", None))

    monkeypatch.setattr(store_migrate, "PostgresStore", FakeStore)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--ensure-schema",
            "--postgres-url",
            "postgresql://db/gpu_fault",
        ],
    )

    store_migrate.main()

    assert calls == [
        ("open", "postgresql://db/gpu_fault", {"hot_state_mode": "legacy"}),
        ("close", None),
    ]
    assert "schema initialization complete" in capsys.readouterr().out


def test_store_migrate_can_backfill_hot_state(monkeypatch, capsys) -> None:
    calls = []

    class FakeStore:
        def __init__(self, url, **kwargs) -> None:
            calls.append(("open", url, kwargs))

        def backfill_hot_state_tables(self):
            return {"gpu_metric_latest": 3}

        def hot_state_migration_status(self):
            return {
                "gpu_metric_latest": {
                    "legacy": 3,
                    "dedicated": 3,
                    "matched_keys": 3,
                    "missing_or_mismatched": 0,
                }
            }

        def close(self) -> None:
            calls.append(("close", None))

    monkeypatch.setattr(store_migrate, "PostgresStore", FakeStore)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--backfill-hot-state",
            "--postgres-url",
            "postgresql://db/gpu_fault",
        ],
    )

    store_migrate.main()

    body = json.loads(capsys.readouterr().out)
    assert body["backfill"]["gpu_metric_latest"] == 3
    assert body["status"]["gpu_metric_latest"]["missing_or_mismatched"] == 0
    assert calls == [
        ("open", "postgresql://db/gpu_fault", {"hot_state_mode": "dual"}),
        ("close", None),
    ]


def test_store_migrate_can_finalize_counter_shards(monkeypatch, capsys) -> None:
    calls = []

    class FakeStore:
        def __init__(self, url, **kwargs) -> None:
            calls.append(("open", url, kwargs))

        def finalize_processor_counter_shards(self):
            return {"mode": "partitioned", "ready": True}

        def close(self) -> None:
            calls.append(("close", None))

    monkeypatch.setattr(store_migrate, "PostgresStore", FakeStore)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--finalize-processor-counter-shards",
            "--postgres-url",
            "postgresql://db/gpu_fault",
        ],
    )

    store_migrate.main()

    assert json.loads(capsys.readouterr().out) == {"mode": "partitioned", "ready": True}
    assert calls == [("open", "postgresql://db/gpu_fault", {}), ("close", None)]


def test_store_migrate_can_restore_legacy_counters(monkeypatch, capsys) -> None:
    class FakeStore:
        def __init__(self, _url, **_kwargs) -> None:
            pass

        def restore_legacy_processor_counters(self):
            return {"mode": "dual", "ready": True}

        def close(self) -> None:
            pass

    monkeypatch.setattr(store_migrate, "PostgresStore", FakeStore)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--restore-legacy-processor-counters",
            "--postgres-url",
            "postgresql://db/gpu_fault",
        ],
    )

    store_migrate.main()

    assert json.loads(capsys.readouterr().out)["mode"] == "dual"


def test_store_migrate_refuses_unsafe_legacy_purge(monkeypatch) -> None:
    class FakeStore:
        def __init__(self, _url, **_kwargs) -> None:
            pass

        def purge_legacy_hot_state(self):
            raise RuntimeError("hot state backfill is incomplete")

        def close(self) -> None:
            pass

    monkeypatch.setattr(store_migrate, "PostgresStore", FakeStore)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-store-migrate",
            "--purge-legacy-hot-state",
            "--postgres-url",
            "postgresql://db/gpu_fault",
        ],
    )

    try:
        store_migrate.main()
    except RuntimeError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("unsafe hot-state purge was accepted")
