#!/usr/bin/env python3
"""Local reproduction of the bloated-store start-burst on the real schema.

The same admission / claim / completion workload runs through ``PostgresStore``
against the queue in three states -- a compact store, the store after one
simulated round and its teardown (every row deleted, a few hundred surviving,
vacuumed twice), and the same store after ``REINDEX`` -- and the report
records per phase the admitted rows, the admission latency percentiles, the
``pg_stat_activity`` wait mix with the statements waiting on
``LWLock:BufferContent``, the relation sizes before and after, and the
queue's ``n_tup_hot_upd`` share. Developer gate for a layout change
(performance acceptance plan §3.3); it does not replace the live steady-state
round. Needs a disposable PostgreSQL: ``HARNESS_URL`` (default a local
container), ``HARNESS_LAYOUT`` ``inline``/``detached`` (the layout the store
is switched to before the phases), ``PROCS``/``PRODUCERS``/``CONSUMERS``/
``DURATION`` for the burst shape, ``REPORT`` for the JSON path.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
os.environ.setdefault("GPU_FAULT_PROCESSOR_QUEUE_STATE_MODE", "dedicated")

URL = os.environ.get("HARNESS_URL", "postgresql://postgres@127.0.0.1:35151/harness")
LAYOUT = os.environ.get("HARNESS_LAYOUT", "detached")
CLUSTERS = 32
NODES = 256
LEASE = timedelta(seconds=120)
TABLES = (
    "gpu_fault_processor_queue",
    "gpu_fault_processor_payloads",
    "gpu_fault_processor_replays",
    "gpu_fault_processor_lanes",
    "gpu_fault_processor_queue_counts",
    "gpu_fault_processor_priority_count_shards",
)
METRICS = (
    "DCGM_FI_DEV_GPU_UTIL",
    "DCGM_FI_DEV_FB_USED",
    "DCGM_FI_DEV_POWER_USAGE",
    "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_DEV_SM_CLOCK",
    "DCGM_FI_DEV_XID_ERRORS",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL",
)


def _body(kind: str, cluster: str, node: str, seq: int) -> tuple[str, dict]:
    stamp = "2026-01-01T00:00:00Z"
    if kind == "metrics":
        samples = [
            {
                "gpu_id": f"GPU-{g:02d}",
                "metric": m,
                "value": round(random.random() * 100, 3),
                "observed_at": stamp,
                "labels": {"node_id": node, "device": f"nvidia{g}"},
            }
            for g in range(8)
            for m in METRICS
        ]
        return "/v1/collector-events/gpu-metrics", {
            "cluster_id": cluster,
            "node_id": node,
            "sequence": seq,
            "observed_at": stamp,
            "samples": samples,
        }
    if kind == "host":
        return "/v1/collector-events/host-telemetry", {
            "cluster_id": cluster,
            "node_id": node,
            "sequence": seq,
            "cpu": [round(random.random(), 4) for _ in range(96)],
            "memory": {"used": 1, "total": 2},
            "disks": [{"dev": f"nvme{i}", "used": random.random()} for i in range(8)],
        }
    if kind == "inventory":
        return "/v1/collector-events/gpu-inventory", {
            "cluster_id": cluster,
            "node_id": node,
            "snapshot_id": f"inv-{seq}",
            "gpus": [
                {
                    "gpu_id": f"GPU-{g}",
                    "serial": f"S{seq}{g}",
                    "vbios": "96.00.89.00.01",
                }
                for g in range(8)
            ],
        }
    return "/v1/gpu-events/xid", {
        "cluster_id": cluster,
        "node_id": node,
        "xid": 79,
        "gpu_id": f"GPU-0{seq % 8}",
        "job_id": f"job-{seq}",
        "attempt_id": f"attempt-{seq}",
        "observed_at": stamp,
        "message": "GPU has fallen off the bus " * 8,
    }


def build_request(tag: str, cluster_index: int, node_index: int, seq: int, kind: str):
    from gpu_fault.processor import ProcessorRequest

    cluster = f"perf-cap-{cluster_index:03d}"
    node = f"node-{tag}-{cluster_index:02d}-{node_index:03d}"
    path, payload = _body(kind, cluster, node, seq)
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return ProcessorRequest.from_http(
        method="POST",
        path=path,
        query="",
        body=raw,
        content_type="application/json",
        cluster_id=cluster,
        parsed_payload=payload,
    )


def kind_for(value: float) -> str:
    if value < 0.45:
        return "metrics"
    if value < 0.75:
        return "host"
    if value < 0.95:
        return "inventory"
    return "xid"


def worker_process(index, tag, producers, consumers, duration, result_queue, start_at):
    import psycopg

    from gpu_fault.store import PostgresStore

    store = PostgresStore(
        URL,
        initialize_schema=False,
        pool_min_size=2,
        pool_max_size=producers + consumers + 4,
        pool_timeout_seconds=60,
    )
    stop = threading.Event()
    drain_stop = threading.Event()
    latencies: list[float] = []
    counts: Counter = Counter()
    lock = threading.Lock()

    def producer(pidx: int) -> None:
        rng = random.Random(index * 1000 + pidx)
        seq = 0
        while time.time() < start_at:
            time.sleep(0.01)
        while not stop.is_set():
            g = index * producers + pidx
            batch = [
                build_request(
                    tag,
                    g % CLUSTERS,
                    g * 16 + ((seq * 10 + i) % 16),
                    seq * 10 + i + index * 10**7 + pidx * 10**5,
                    kind_for(rng.random()),
                )
                for i in range(10)
            ]
            seq += 1
            started = time.perf_counter()
            try:
                results = store.try_enqueue_processor_requests_batch(
                    batch,
                    max_depth=200000,
                    max_cluster_depth=20000,
                    reserved_fault_depth=50000,
                    reserved_cluster_fault_depth=1000,
                    global_admission_guard=256,
                )
                with lock:
                    latencies.append(time.perf_counter() - started)
                    counts["admitted"] += sum(
                        1
                        for item, reason in results
                        if item is not None and reason is None
                    )
                    counts["coalesced"] += sum(
                        1 for _, reason in results if reason == "coalesced"
                    )
                    counts["rejected"] += sum(1 for item, _ in results if item is None)
            except Exception as exc:  # noqa: BLE001 - counted
                with lock:
                    counts[f"error:{type(exc).__name__}"] += 1
                time.sleep(0.05)

    def consumer(cidx: int) -> None:
        owner = f"harness-{index}-{cidx}"
        while time.time() < start_at:
            time.sleep(0.01)
        while not drain_stop.is_set():
            try:
                claimed = store.claim_active_processor_requests(
                    owner, now=datetime.now(timezone.utc), lease_duration=LEASE, limit=8
                )
                if not claimed:
                    time.sleep(0.02 if stop.is_set() else 0.05)
                    continue
                store.complete_active_processor_requests_batch(
                    [
                        {
                            "request_id": entry.request_id,
                            "owner_id": owner,
                            "lane_epoch": entry.leader_epoch,
                            "lease_token": entry.lease_token,
                            "response_status": 200,
                            "response_content_type": "application/json",
                            "response_body_base64": "eyJvayI6dHJ1ZX0=",
                        }
                        for entry in claimed
                    ]
                )
                with lock:
                    counts["completed"] += len(claimed)
            except Exception as exc:  # noqa: BLE001 - counted
                with lock:
                    counts[f"cerror:{type(exc).__name__}"] += 1
                time.sleep(0.05)

    threads = [
        threading.Thread(target=producer, args=(p,), daemon=True)
        for p in range(producers)
    ]
    threads += [
        threading.Thread(target=consumer, args=(c,), daemon=True)
        for c in range(consumers)
    ]
    for thread in threads:
        thread.start()
    time.sleep(max(0.0, start_at - time.time()) + duration)
    stop.set()
    deadline = time.time() + 120
    with psycopg.connect(URL, autocommit=True) as connection:
        while time.time() < deadline:
            pending = connection.execute(
                "SELECT count(*) FROM gpu_fault_processor_queue "
                "WHERE status IN ('PENDING','LEASED')"
            ).fetchone()[0]
            if pending == 0:
                break
            time.sleep(1.0)
    drain_stop.set()
    for thread in threads:
        thread.join(timeout=30)
    store.close()
    ordered = sorted(latencies)

    def percentile(p: float) -> float | None:
        return ordered[int((len(ordered) - 1) * p)] * 1000 if ordered else None

    result_queue.put(
        {
            "proc": index,
            "counts": dict(counts),
            "batches": len(ordered),
            "admit_p50_ms": percentile(0.5),
            "admit_p99_ms": percentile(0.99),
            "admit_max_ms": percentile(1.0),
        }
    )


def sizes(cursor) -> dict:
    out: dict = {}
    for table in TABLES:
        cursor.execute("SELECT to_regclass(%s)", (table,))
        if cursor.fetchone()[0] is None:
            continue
        cursor.execute(
            """
            WITH members AS (
                SELECT c.oid FROM pg_class c WHERE c.oid = to_regclass(%s) AND c.relkind <> 'p'
                UNION ALL
                SELECT i.inhrelid FROM pg_inherits i WHERE i.inhparent = to_regclass(%s)
            )
            SELECT coalesce(sum(c.relpages), 0), coalesce(sum(s.n_live_tup), 0),
                   coalesce(sum(s.n_dead_tup), 0),
                   coalesce(sum(pg_relation_size(c.oid)), 0),
                   coalesce(sum(pg_indexes_size(c.oid)), 0),
                   coalesce(sum(pg_total_relation_size(c.reltoastrelid)), 0),
                   coalesce(sum(s.n_tup_upd), 0), coalesce(sum(s.n_tup_hot_upd), 0)
            FROM members m JOIN pg_class c ON c.oid = m.oid
            LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
            """,
            (table, table),
        )
        row = cursor.fetchone()
        out[table] = {
            "relpages": int(row[0]),
            "live": int(row[1]),
            "dead": int(row[2]),
            "heap_kb": int(row[3]) // 1024,
            "idx_kb": int(row[4]) // 1024,
            "toast_kb": int(row[5]) // 1024,
            "upd": int(row[6]),
            "hot_upd": int(row[7]),
        }
    cursor.execute(
        "SELECT indexrelname, pg_relation_size(indexrelid) FROM pg_stat_user_indexes "
        "WHERE relname='gpu_fault_processor_queue' ORDER BY 1"
    )
    out["queue_indexes_kb"] = {name: size // 1024 for name, size in cursor.fetchall()}
    return out


class Sampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.waits: Counter = Counter()
        self.by_statement: Counter = Counter()
        self.buffer_statements: Counter = Counter()
        self.samples = 0
        self.peak = 0

    def run(self) -> None:
        import psycopg

        with psycopg.connect(URL, autocommit=True) as connection:
            while not self.stop.is_set():
                rows = connection.execute(
                    """
                    SELECT coalesce(wait_event_type,'CPU'), coalesce(wait_event,''),
                           left(regexp_replace(query,'\\s+',' ','g'),70)
                    FROM pg_stat_activity
                    WHERE datname=current_database() AND state='active'
                      AND pid<>pg_backend_pid() AND backend_type='client backend'
                    """
                ).fetchall()
                self.samples += 1
                self.peak = max(self.peak, len(rows))
                for wait_type, wait, statement in rows:
                    self.waits[f"{wait_type}:{wait}"] += 1
                    self.by_statement[statement] += 1
                    if wait == "BufferContent":
                        self.buffer_statements[statement] += 1
                time.sleep(0.2)

    def report(self) -> dict:
        total = sum(self.waits.values()) or 1
        return {
            "samples": self.samples,
            "active_rows": total,
            "peak_active": self.peak,
            "waits_pct": {
                k: round(100 * v / total, 1) for k, v in self.waits.most_common(8)
            },
            "top_statements": dict(self.by_statement.most_common(8)),
            "buffercontent_by_statement": dict(self.buffer_statements.most_common(8)),
        }


def teardown(cursor, store) -> None:
    """The perf teardown's shape: every queue row, lane and replay bucket gone,
    then two vacuum passes so the B-trees recycle what they can."""

    cursor.execute("DELETE FROM gpu_fault_processor_queue")
    cursor.execute("DELETE FROM gpu_fault_processor_payloads")
    cursor.execute("DELETE FROM gpu_fault_processor_lanes")
    cursor.execute("DELETE FROM gpu_fault_objects WHERE kind='processor_request'")
    cursor.execute(
        "DELETE FROM gpu_fault_processor_queue_counts WHERE cluster_id LIKE 'perf-%'"
    )
    cursor.execute(
        "DELETE FROM gpu_fault_processor_priority_count_shards WHERE cluster_id LIKE 'perf-%'"
    )
    store.cleanup_completed_processor_requests(
        older_than=datetime.now(timezone.utc) + timedelta(hours=1), limit=10**6
    )
    for _ in range(2):
        for table in TABLES:
            cursor.execute("SELECT to_regclass(%s)", (table,))
            if cursor.fetchone()[0] is not None:
                cursor.execute(f"VACUUM (ANALYZE) {table}")
        time.sleep(0.5)


def prepare(cursor, action: str) -> None:
    if action == "none":
        return
    if action == "reindex_all":
        cursor.execute(
            "SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
            "WHERE i.indrelid IN ('gpu_fault_processor_queue'::regclass, "
            "'gpu_fault_processor_payloads'::regclass)"
        )
        for (name,) in cursor.fetchall():
            cursor.execute(f"REINDEX INDEX {name}")
        for table in (
            "gpu_fault_processor_lanes",
            "gpu_fault_processor_queue_counts",
            "gpu_fault_processor_priority_count_shards",
        ):
            cursor.execute(f"VACUUM (FULL, ANALYZE) {table}")
        cursor.execute("VACUUM (ANALYZE) gpu_fault_processor_queue")


def main() -> int:
    import psycopg

    from gpu_fault.store import PostgresStore
    from gpu_fault.store.postgres import hot_layout

    phases = json.loads(
        os.environ.get(
            "PHASES",
            '[["fresh","none"],["bloated","none"],["all_rebuilt","reindex_all"]]',
        )
    )
    procs = int(os.environ.get("PROCS", "20"))
    producers = int(os.environ.get("PRODUCERS", "8"))
    consumers = int(os.environ.get("CONSUMERS", "3"))
    duration = float(os.environ.get("DURATION", "30"))
    report: dict = {
        "config": {
            "procs": procs,
            "producers": producers,
            "consumers": consumers,
            "duration": duration,
            "layout": LAYOUT,
        },
        "phases": [],
    }
    PostgresStore(URL).close()
    with psycopg.connect(URL, autocommit=True) as connection:
        if LAYOUT == "detached":
            hot_layout.switch_to_detached(connection)
        else:
            connection.execute(
                "UPDATE gpu_fault_hot_layout_mode SET mode='inline' WHERE singleton"
            )
    store = PostgresStore(
        URL, initialize_schema=False, pool_min_size=1, pool_max_size=4
    )
    with psycopg.connect(URL, autocommit=True) as connection:
        cursor = connection.cursor()
        teardown(cursor, store)
        for name, action in phases:
            prepare(cursor, action)
            before = sizes(cursor)
            sampler = Sampler()
            sampler.start()
            queue: mp.Queue = mp.Queue()
            start_at = time.time() + 3
            processes = [
                mp.Process(
                    target=worker_process,
                    args=(i, name, producers, consumers, duration, queue, start_at),
                )
                for i in range(procs)
            ]
            started = time.time()
            for process in processes:
                process.start()
            results = [queue.get() for _ in processes]
            for process in processes:
                process.join()
            wall = time.time() - started - 3
            sampler.stop.set()
            sampler.join(timeout=5)
            admitted = sum(r["counts"].get("admitted", 0) for r in results)
            coalesced = sum(r["counts"].get("coalesced", 0) for r in results)
            completed = sum(r["counts"].get("completed", 0) for r in results)
            after = sizes(cursor)
            phase = {
                "name": name,
                "prep": action,
                "admitted_rows": admitted,
                "coalesced": coalesced,
                "completed": completed,
                "admit_rows_per_s": round((admitted + coalesced) / duration, 1),
                "wall_incl_drain_s": round(wall, 1),
                "admit_p50_ms": round(
                    statistics.median([r["admit_p50_ms"] or 0 for r in results]), 1
                ),
                "admit_p99_ms": round(
                    max((r["admit_p99_ms"] or 0) for r in results), 1
                ),
                "errors": {
                    k: v
                    for r in results
                    for k, v in r["counts"].items()
                    if "error" in k
                },
                "sampler": sampler.report(),
                "sizes_before": before,
                "sizes_after": after,
                "queue_hot_update_share": (
                    round(
                        (
                            after["gpu_fault_processor_queue"]["hot_upd"]
                            - before["gpu_fault_processor_queue"]["hot_upd"]
                        )
                        / max(
                            1,
                            after["gpu_fault_processor_queue"]["upd"]
                            - before["gpu_fault_processor_queue"]["upd"],
                        ),
                        3,
                    )
                ),
            }
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in phase.items()
                        if k not in ("sizes_before", "sizes_after")
                    }
                ),
                flush=True,
            )
            report["phases"].append(phase)
            teardown(cursor, store)
    store.close()
    out = Path(os.environ.get("REPORT", "/tmp/store-layout-bloat-harness.json"))
    out.write_text(json.dumps(report, indent=1, default=str))
    print("written", out)
    return 0


if __name__ == "__main__":
    mp.set_start_method("fork")
    raise SystemExit(main())
