"""Per-tier completion latency of a capacity run's processor requests.

``processor-priority-latency.json`` was empty in every run on a schema v23
site: it read the COMPLETED rows of ``gpu_fault_processor_queue``, and in the
detached hot layout a completion leaves the queue -- the request moves to the
``gpu_fault_processor_replays`` partitions (``ddl_processor_layout``). This
probe reads where the active layout (``gpu_fault_hot_layout_mode``) keeps
completed requests, and both while a layout switch leaves rows in each: the
queue's COMPLETED rows with their stored ``priority`` (the inline layout, and
rows completed before a switch to detached) and the replay rows.

A replay row stores no priority. The tier is a function of the path except on
the edge-filtered channels, where it also reads the body's
``collection_errors`` and ``edge_filter_reasons``
(``ProcessorChannel.is_routine_payload``). The replay statement extracts those
two fields in SQL and groups by them, so no body leaves the database; a body
PostgreSQL cannot read as a JSON object (``NaN``, a NUL byte, an empty or a
non-object body) comes back whole and is classified on the full body. Every
group's tier is the product's own ``ProcessorRequest.queue_priority()``, not
a copy of its rules.

Latency is completion minus creation in milliseconds -- ``updated_at -
created_at`` for a queue row, ``completed_at - envelope.created_at`` for a
replay (``completed_at`` is the request's ``updated_at`` at completion) -- and
the percentiles interpolate like ``percentile_cont``. Rows are scoped to the
run's synthetic clusters and to requests created since ``since_epoch``. The
product keeps completed requests for ``completedRetentionSeconds`` (600 s by
default), so the document states the completion interval each source covered.

Shipped to a control-plane Pod as source and run with the product's
interpreter (``python -``), like ``regional_capacity_data``.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

PERCENTILES = (("p50_ms", 0.50), ("p95_ms", 0.95), ("p99_ms", 0.99))
COMPLETED = "COMPLETED"

#: A control plane older than schema v23 has neither table: queue only.
LAYOUT_TABLES_SQL = """
SELECT to_regclass('gpu_fault_hot_layout_mode') IS NOT NULL,
       to_regclass('gpu_fault_processor_replays') IS NOT NULL
"""
LAYOUT_MODE_SQL = "SELECT mode FROM gpu_fault_hot_layout_mode WHERE singleton"
QUEUE_STATUS_SQL = """
SELECT priority, status, count(*)
FROM gpu_fault_processor_queue
WHERE cluster_id LIKE %(prefix)s
  AND created_at >= to_timestamp(%(since)s)
GROUP BY priority, status
"""
QUEUE_COMPLETED_SQL = """
SELECT priority, cluster_id,
       array_agg(extract(epoch FROM updated_at - created_at) * 1000),
       min(extract(epoch FROM updated_at)), max(extract(epoch FROM updated_at))
FROM gpu_fault_processor_queue
WHERE cluster_id LIKE %(prefix)s
  AND status = 'COMPLETED'
  AND created_at >= to_timestamp(%(since)s)
GROUP BY priority, cluster_id
"""
#: ``parsed`` is an optimisation fence (``OFFSET 0``) so a body is decoded
#: and parsed once per row; LATIN1 maps every byte but NUL to a character, so
#: the decode cannot fail and the classification, which compares the reasons
#: with ASCII names and prefixes, reads the same values UTF-8 would give.
REPLAYS_SQL = r"""
SELECT r.cluster_id, r.envelope->>'path', edge.fields,
       CASE WHEN edge.unreadable THEN coalesce(r.body, '\x'::bytea) END,
       array_agg(
           extract(epoch FROM r.completed_at
                   - (r.envelope->>'created_at')::timestamptz) * 1000
       ),
       min(extract(epoch FROM r.completed_at)),
       max(extract(epoch FROM r.completed_at))
FROM gpu_fault_processor_replays AS r
CROSS JOIN LATERAL (
    SELECT
        CASE WHEN jsonb_typeof(parsed.doc) = 'object' THEN jsonb_build_object(
            'collection_errors', parsed.doc -> 'collection_errors',
            'edge_filter_reasons', parsed.doc -> 'edge_filter_reasons'
        ) END AS fields,
        parsed.edge AND jsonb_typeof(parsed.doc) IS DISTINCT FROM 'object'
            AS unreadable
    FROM (
        SELECT
            r.envelope->>'path' = ANY(%(edge_paths)s) AS edge,
            CASE
                WHEN NOT r.envelope->>'path' = ANY(%(edge_paths)s) THEN NULL
                WHEN r.body IS NULL OR position('\x00'::bytea IN r.body) > 0
                    THEN NULL
                WHEN pg_input_is_valid(convert_from(r.body, 'LATIN1'), 'jsonb')
                    THEN convert_from(r.body, 'LATIN1')::jsonb
            END AS doc
        OFFSET 0
    ) AS parsed
) AS edge
WHERE r.cluster_id LIKE %(prefix)s
  AND r.completed_at >= to_timestamp(%(since)s)
  AND (r.envelope->>'created_at')::timestamptz >= to_timestamp(%(since)s)
GROUP BY 1, 2, 3, 4
"""


def percentile(ordered: list[float], fraction: float) -> float:
    """``percentile_cont`` over an ascending, non-empty list."""

    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def edge_filtered_paths() -> list[str]:
    from gpu_fault.channel_registry import CHANNEL_REGISTRY

    return sorted(
        path for path, channel in CHANNEL_REGISTRY.items() if channel.edge_filtered
    )


class Tiers:
    """The product's tier for a path and the body a group stands for."""

    def __init__(self) -> None:
        from gpu_fault.processor.models import ProcessorRequest

        self._request = ProcessorRequest
        self._known: dict[tuple[str, bytes], int] = {}

    def __call__(self, path: str, body: bytes) -> int:
        key = (path, body)
        if key not in self._known:
            request = self._request(
                method="POST",
                path=path,
                body_base64=base64.b64encode(body).decode("ascii"),
            )
            self._known[key] = int(request.queue_priority())
        return self._known[key]


def replay_groups(
    rows: Iterable[tuple[Any, ...]], tier: Callable[[str, bytes], int]
) -> tuple[list[tuple[int, str, list[float]]], int]:
    """(tier, cluster, latencies) per replay group, and how many rows were
    classified on their full body."""

    groups = []
    unreadable = 0
    for cluster, path, fields, raw, values, _first, _last in rows:
        if raw is not None:
            body = bytes(raw)
            unreadable += len(values)
        elif fields is not None:
            body = json.dumps(fields, sort_keys=True).encode()
        else:
            body = b""
        groups.append((tier(path or "", body), cluster, [float(v) for v in values]))
    return groups, unreadable


def latency_sections(
    groups: Iterable[tuple[int, str, list[float]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``completion_latency`` per tier and ``cluster_fairness`` (the spread of
    the per-cluster p99) per tier."""

    by_tier: dict[int, list[float]] = {}
    by_cluster: dict[tuple[int, str], list[float]] = {}
    for tier, cluster, values in groups:
        by_tier.setdefault(tier, []).extend(values)
        by_cluster.setdefault((tier, cluster), []).extend(values)
    latency = {}
    for tier, values in sorted(by_tier.items()):
        ordered = sorted(values)
        latency[str(tier)] = {
            "count": len(ordered),
            **{
                name: round(percentile(ordered, fraction), 3)
                for name, fraction in PERCENTILES
            },
            "max_ms": round(ordered[-1], 3),
        }
    p99s: dict[int, list[float]] = {}
    for (tier, _cluster), values in sorted(by_cluster.items()):
        p99s.setdefault(tier, []).append(percentile(sorted(values), 0.99))
    fairness = {
        str(tier): {
            "cluster_p99_min_ms": round(min(values), 3),
            "cluster_p99_max_ms": round(max(values), 3),
            "cluster_p99_mean_ms": round(sum(values) / len(values), 3),
        }
        for tier, values in sorted(p99s.items())
    }
    return latency, fairness


def _covered(rows: list[tuple[Any, ...]]) -> dict[str, Any]:
    firsts = [float(row[-2]) for row in rows if row[-2] is not None]
    lasts = [float(row[-1]) for row in rows if row[-1] is not None]
    return {
        "completed": sum(len(row[-3]) for row in rows),
        "first_completed_at_epoch": round(min(firsts), 3) if firsts else None,
        "last_completed_at_epoch": round(max(lasts), 3) if lasts else None,
    }


def measure(connection: Any, *, prefix: str, since_epoch: float) -> dict[str, Any]:
    """The latency document over one open connection (autocommit reads)."""

    parameters = {"prefix": prefix + "%", "since": float(since_epoch)}
    cursor = connection.cursor()
    cursor.execute(LAYOUT_TABLES_SQL)
    has_mode, has_replays = cursor.fetchone()
    mode = None
    if has_mode:
        cursor.execute(LAYOUT_MODE_SQL)
        row = cursor.fetchone()
        mode = row[0] if row else None
    statuses: dict[str, dict[str, int]] = {}
    cursor.execute(QUEUE_STATUS_SQL, parameters)
    for priority, status, count in cursor.fetchall():
        statuses.setdefault(str(priority), {})[str(status)] = int(count)
    cursor.execute(QUEUE_COMPLETED_SQL, parameters)
    queue_rows = cursor.fetchall()
    groups = [
        (int(priority), cluster, [float(value) for value in values])
        for priority, cluster, values, _first, _last in queue_rows
    ]
    sources: dict[str, Any] = {"queue": _covered(queue_rows)}
    if has_replays:
        cursor.execute(REPLAYS_SQL, {**parameters, "edge_paths": edge_filtered_paths()})
        replay_rows = cursor.fetchall()
        replayed, unreadable = replay_groups(replay_rows, Tiers())
        for tier, _cluster, values in replayed:
            counts = statuses.setdefault(str(tier), {})
            counts[COMPLETED] = counts.get(COMPLETED, 0) + len(values)
        groups.extend(replayed)
        sources["replays"] = {
            **_covered(replay_rows),
            "classified_on_full_body": unreadable,
        }
    latency, fairness = latency_sections(groups)
    return {
        "layout_mode": mode,
        "completed_source": "replays" if mode == "detached" else "queue",
        "cluster_prefix": prefix,
        "since_epoch": float(since_epoch),
        "sources": sources,
        "status_counts": {key: statuses[key] for key in sorted(statuses, key=int)},
        "completion_latency": latency,
        "cluster_fairness": fairness,
    }


def invoke(
    control: Callable[..., str],
    pods: list[str],
    *,
    cluster_prefix: str,
    since_epoch: float = 0.0,
) -> dict[str, Any]:
    """Run :func:`measure` in the first api-ha Pod; an error is the document."""

    if not pods:
        return {"error": "no api pod available"}
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.e2e.regional.regional_live_fixture import component_python

    arguments = {"cluster_prefix": cluster_prefix, "since_epoch": since_epoch}
    try:
        output = control(
            "exec",
            "-i",
            pods[0],
            "--",
            component_python("cpu"),
            "-",
            json.dumps(arguments),
            stdin=Path(__file__).read_bytes(),
            timeout=300,
        )
    except Exception as exc:  # noqa: BLE001 - a measurement, not the verdict
        return {"error": str(exc)[-2000:]}
    for line in reversed(output.splitlines()):
        if line.startswith("{"):
            try:
                return dict(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"error": output[-2000:]}


def run_request(arguments: dict[str, Any]) -> dict[str, Any]:
    import psycopg

    from gpu_fault.store.postgres.pool import StoreCredentials

    path = os.getenv("GPU_FAULT_STORE_URL_FILE")
    credentials = StoreCredentials(os.getenv("GPU_FAULT_STORE_URL", ""), path=path)
    url = credentials.conninfo()
    if not url or (path and credentials.source != "file"):
        raise RuntimeError("current capacity database credentials are unavailable")
    with psycopg.connect(url, connect_timeout=10, autocommit=True) as connection:
        return measure(
            connection,
            prefix=str(arguments["cluster_prefix"]),
            since_epoch=float(arguments.get("since_epoch") or 0.0),
        )


if __name__ == "__main__":
    print(json.dumps(run_request(json.loads(sys.argv[1])), sort_keys=True))
