from __future__ import annotations

from contextlib import nullcontext

from gpu_fault.store.shared.record_models import record_models
from tests._builders import fault_incident, workflow_request

INCIDENT = fault_incident("migration-incident", "migration-event")
WORKFLOW = workflow_request("migration-workflow", INCIDENT.incident_id)
OBJECTS = [
    ("incident", INCIDENT.incident_id, INCIDENT.model_dump_json()),
    ("workflow", WORKFLOW.request_id, WORKFLOW.model_dump_json()),
]
LINKS = [("event-incident", "migration-event", INCIDENT.incident_id)]


class Cursor:
    def __init__(self, store) -> None:
        self.store = store
        self.statement = ""

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False

    def execute(self, statement, parameters=None):
        self.statement = " ".join(statement.split())
        self.store.events.append(("execute", self.store.url, self.statement))

    def fetchall(self):
        return self.store.objects if "control_records" in self.statement else LINKS

    def fetchone(self):
        return (self.store.existing,)

    def executemany(self, statement, rows):
        self.store.events.append(("write", self.store.url, " ".join(statement.split())))
        self.store.writes.append((" ".join(statement.split()), list(rows)))


class Database:
    def __init__(self, store) -> None:
        self.store = store

    def transaction(self):
        self.store.events.append(("transaction", self.store.url))
        return nullcontext()

    def cursor(self):
        return Cursor(self.store)


class Store:
    def __init__(
        self, url, *, events, existing=0, invalid=False, close_error=False, **options
    ):
        self.url = url
        self.events = events
        self.options = options
        self.existing = existing
        self.invalid = invalid
        self.invalid_decode = False
        self.close_error = close_error
        self.closed = False
        self.writes = []
        self.objects = list(OBJECTS)
        self.model_types = record_models()
        self.hot_status = {
            kind: {
                "legacy": 0,
                "dedicated": 0,
                "matched_keys": 0,
                "missing_or_mismatched": 0,
            }
            for kind in (
                "gpu_metric_latest",
                "gpu_metrics_batch",
                "attempt_observation",
                "training_progress",
            )
        }
        self._db = Database(self)
        self.events.append(("open", url, options))

    def _decode(self, kind, payload):
        self.events.append(("decode", self.url, kind))
        if self.invalid or self.invalid_decode:
            raise ValueError("invalid source record")
        return self.model_types[kind].model_validate_json(payload)

    def close(self):
        self.closed = True
        self.events.append(("close", self.url))
        if self.close_error:
            raise RuntimeError("destination close failed")

    def __getattr__(self, name):
        values = {
            "hot_state_migration_status": self.hot_status,
            "processor_queue_state_status": {"ready": True},
            "processor_queue_count_status": {"count": 0},
            "processor_counter_mode": "partitioned",
            "backfill_hot_state_tables": {"backfilled": 2},
            "backfill_processor_queue_state_columns": 2,
            "finalize_processor_counter_shards": {"ready": True},
            "restore_legacy_processor_counters": {"ready": True},
            "purge_legacy_hot_state": {"deleted": 2},
        }
        if name not in values:
            raise AttributeError(name)

        def call():
            self.events.append(("method", self.url, name))
            if self.invalid:
                raise ValueError("operation refused")
            return values[name]

        return call
