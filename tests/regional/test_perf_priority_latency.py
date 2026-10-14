"""The priority latency probe reads where the active layout keeps completions.

``processor-priority-latency.json`` was empty in every run on the detached
layout: the probe read COMPLETED queue rows, and a detached completion leaves
the queue for the replay partitions. The SQL against both layouts runs in
``tests/store/test_postgres_hot_layout.py``; these tests pin the reduction,
the shipping into the Pod and the classification shortcut the replay
statement relies on (only two body fields decide an edge-filtered tier).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.channel_registry import CHANNEL_REGISTRY
from scripts.perf import regional_capacity_priority_latency as latency
from scripts.perf import regional_capacity_suite as capacity

EDGE_FIELDS = {"collection_errors", "edge_filter_reasons"}


def test_percentiles_interpolate_like_percentile_cont() -> None:
    ordered = [10.0, 20.0, 30.0, 40.0]

    assert latency.percentile(ordered, 0.50) == 25.0
    assert latency.percentile(ordered, 0.99) == pytest.approx(39.7)
    assert latency.percentile([7.0], 0.95) == 7.0


def test_latency_merges_both_sources_per_tier_and_spreads_per_cluster() -> None:
    groups = [
        (10, "perf-cap-000", [10.0, 20.0]),  # queue rows
        (10, "perf-cap-000", [30.0]),  # replay rows of the same tier
        (10, "perf-cap-001", [100.0]),
        (100, "perf-cap-001", [5.0]),
    ]

    completion, fairness = latency.latency_sections(groups)

    assert completion["10"] == {
        "count": 4,
        "p50_ms": 25.0,
        "p95_ms": 89.5,
        "p99_ms": 97.9,
        "max_ms": 100.0,
    }, "one distribution per tier, whichever table a row came from"
    assert completion["100"]["count"] == 1
    assert fairness["10"] == {
        "cluster_p99_min_ms": 29.8,
        "cluster_p99_max_ms": 100.0,
        "cluster_p99_mean_ms": 64.9,
    }


def test_a_replay_group_is_classified_on_what_the_statement_returned() -> None:
    seen: list[tuple[str, bytes]] = []

    def tier(path: str, body: bytes) -> int:
        seen.append((path, body))
        return 7

    fields = {"collection_errors": None, "edge_filter_reasons": ["threshold:temp"]}
    rows = [
        ("perf-cap-000", "/v1/gpu-events/xid", None, None, [1, 2], 1.0, 2.0),
        ("perf-cap-000", "/v1/collector-events/gpu-metrics", fields, None, [3], 1, 1),
        ("perf-cap-001", "/v1/collector-events/node-logs", None, b"{NaN", [4], 1, 1),
    ]

    groups, unreadable = latency.replay_groups(rows, tier)

    assert groups == [
        (7, "perf-cap-000", [1.0, 2.0]),
        (7, "perf-cap-000", [3.0]),
        (7, "perf-cap-001", [4.0]),
    ]
    assert seen == [
        ("/v1/gpu-events/xid", b""),
        (
            "/v1/collector-events/gpu-metrics",
            json.dumps(fields, sort_keys=True).encode(),
        ),
        ("/v1/collector-events/node-logs", b"{NaN"),
    ], "path only, the two extracted fields, or the whole unreadable body"
    assert unreadable == 1


class Reading(dict[str, Any]):
    """A payload that records which fields the tier rule reads."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.read: set[str] = set()

    def get(self, key: str, default: Any = None) -> Any:
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self.read.add(str(key))
        return super().__contains__(key)


PAYLOADS = [
    {"node_id": "n", "gpus": [{"temp": 40}], "edge_filter_reasons": ["baseline"]},
    {"node_id": "n", "edge_filter_reasons": ["baseline:boot", "health-summary"]},
    {"node_id": "n", "edge_filter_reasons": ["threshold:temperature"]},
    {"node_id": "n", "edge_filter_reasons": [], "collection_errors": ["dcgm down"]},
    {"node_id": "n", "lines": ["kernel: ok"]},
    {"node_id": "n", "edge_filter_reasons": "not-a-list"},
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_an_edge_filtered_tier_reads_only_the_two_fields_the_statement_extracts(
    payload: dict[str, Any],
) -> None:
    tiers = latency.Tiers()
    for path in latency.edge_filtered_paths():
        reading = Reading(payload)
        CHANNEL_REGISTRY[path].priority(reading)
        assert reading.read <= EDGE_FIELDS, (
            f"{path} reads {sorted(reading.read - EDGE_FIELDS)}: extract them in "
            "REPLAYS_SQL too"
        )
        extracted = {field: payload.get(field) for field in sorted(EDGE_FIELDS)}
        assert tiers(path, json.dumps(extracted).encode()) == tiers(
            path, json.dumps(payload).encode()
        ), path


def test_a_body_postgresql_cannot_parse_is_classified_like_the_product_does() -> None:
    tiers = latency.Tiers()
    host = "/v1/collector-events/host-telemetry"

    assert tiers(host, b'{"node_id": "n", "v": NaN}') == 100, (
        "Python reads NaN: an unfiltered batch stays routine"
    )
    assert tiers(host, b"") == 50, "no readable body is nobody's routine"
    assert tiers(host, b"[1, 2]") == 50
    assert tiers("/v1/gpu-events/xid", b"") == 10
    assert tiers("/v1/attempts/terminal", b"") == 0


def test_the_probe_runs_in_an_api_pod_with_the_product_interpreter() -> None:
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def control(*args: str, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return 'warning: noise\n{"layout_mode": "detached"}\n'

    document = latency.invoke(
        control, ["gpu-fault-api-ha-0"], cluster_prefix="perf-cap-", since_epoch=12.5
    )

    assert document == {"layout_mode": "detached"}
    ((args, kwargs),) = calls
    assert args == (
        "exec",
        "-i",
        "gpu-fault-api-ha-0",
        "--",
        "/opt/gpu-fault/control-plane/bin/python",
        "-",
        json.dumps({"cluster_prefix": "perf-cap-", "since_epoch": 12.5}),
    )
    assert kwargs["stdin"] == Path(latency.__file__).read_bytes()
    assert latency.invoke(control, [], cluster_prefix="perf-cap-") == {
        "error": "no api pod available"
    }

    def failing(*_args: str, **_kwargs: Any) -> str:
        raise RuntimeError("command failed (1): exec")

    assert latency.invoke(failing, ["api"], cluster_prefix="p") == {
        "error": "command failed (1): exec"
    }, "a failed probe is recorded, not raised into the run"


def test_the_shipped_module_needs_only_the_standard_library_at_import() -> None:
    tree = ast.parse(Path(latency.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }

    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported


def test_the_suite_scopes_the_probe_to_requests_since_the_load_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(capacity, "control_pods", lambda: ["api-ha-0", "worker-0"])
    monkeypatch.setattr(
        capacity,
        "_processor_priority_latency",
        lambda _control, pods, **kwargs: calls.append({"pods": pods, **kwargs}) or {},
    )

    capacity.processor_priority_latency(1234.5)

    assert calls == [
        {"pods": ["api-ha-0"], "cluster_prefix": "perf-cap-", "since_epoch": 1234.5}
    ]
