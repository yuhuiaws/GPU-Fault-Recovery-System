from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.collectors.host.collector import HostTelemetryCollector
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.host_health import NodeHealthPolicy
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from scripts.e2e.regional import collector_inventory_evidence as evidence
from scripts.e2e.regional import run_collector_acceptance as runner
from scripts.e2e.regional import run_collector_destructive as destructive
from scripts.e2e.regional.probes import collector_node_probe as probe

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)
ENV = {
    "GPU_FAULT_EXPECTED_GPU_COUNT": "8",
    "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT": "4",
    "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
    "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
    "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES": "2",
}


def inventory_proof() -> dict[str, Any]:
    return {
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "captured_at": NOW.isoformat(),
        "metrics": [
            {
                "cluster_id": "cluster-a",
                "node_id": "node-a",
                "name": f"{resource}_inventory_{suffix}",
                "value": count,
                "device": None,
                "observed_at": NOW.isoformat(),
                "labels": {
                    "expected_count": str(count),
                    "observed_count": str(count),
                    "consecutive_mismatch_samples": "0",
                    "missing_count": "0",
                    "excess_count": "0",
                },
            }
            for resource, count in (("gpu", 8), ("efa", 4))
            for suffix in ("expected_count", "active_count")
        ],
        "signals": {
            f"{resource}_inventory_mismatch": {
                "signal_key": f"cluster-a/node-a/{resource}_inventory_mismatch/node",
                "active": False,
                "observed_at": NOW.isoformat(),
                "clock_at": NOW.isoformat(),
            }
            for resource in ("gpu", "efa")
        },
        "statuses": [
            {
                "cluster_id": "cluster-a",
                "node_id": "node-a",
                "collector": "HOST_TELEMETRY",
                "batch_id": "host-node-a-1",
                "observed_at": NOW.isoformat(),
                "ingested_at": NOW.isoformat(),
                "errors": [],
            }
        ],
    }


def inventory_errors(proof: dict[str, Any], *, age: int = 330) -> list[str]:
    return evidence.inventory_state_errors(
        proof,
        cluster_id="cluster-a",
        node="node-a",
        expected_gpu=8,
        expected_efa=4,
        max_age_seconds=age,
    )


@pytest.mark.parametrize(
    "value", [None, 0, "", "invalid", "2026-09-15T00:00:00", "2026-09-15T00:00:00Z"]
)
def test_collector_evidence_requires_timezone_aware_timestamp(value: Any) -> None:
    assert (evidence.evidence_time(value) is not None) == str(value).endswith("Z")


def test_scoped_inventory_requires_explicit_fresh_health_not_just_equal_counts() -> (
    None
):
    proof = inventory_proof()
    assert inventory_errors(proof) == []
    assert inventory_errors(proof, age=0), (
        "a zero freshness budget must not authorize inventory evidence"
    )
    proof["signals"]["efa_inventory_mismatch"]["active"] = True
    assert inventory_errors(proof) == [
        "efa_inventory_mismatch: absence of a continuing finding is unproven"
    ]


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("root", "cluster_id", "another-cluster"),
        ("root", "node_id", "another-node"),
        ("root", "captured_at", None),
        ("root", "metrics", None),
        ("root", "metrics", [None]),
        ("root", "metrics", []),
        ("root", "signals", None),
        ("root", "signals", {}),
        ("root", "statuses", None),
        ("root", "statuses", []),
        ("root", "statuses", [None]),
        ("root", "statuses", [{}, {}]),
        ("status", "cluster_id", "another-cluster"),
        ("status", "node_id", "another-node"),
        ("status", "collector", "GPU_METRICS"),
        ("status", "batch_id", "kubernetes-resource-1"),
        ("status", "errors", ["collector failed"]),
        ("status", "ingested_at", (NOW - timedelta(seconds=331)).isoformat()),
        ("status", "observed_at", (NOW + timedelta(seconds=31)).isoformat()),
        ("metric", "cluster_id", "another-cluster"),
        ("metric", "node_id", "another-node"),
        ("metric", "device", "unrelated-device"),
        ("metric", "value", None),
        ("metric", "value", True),
        ("metric", "value", float("nan")),
        ("metric", "value", 7),
        ("metric", "observed_at", None),
        ("metric", "labels", {}),
        ("metric", "labels", {"expected_count": "8", "observed_count": "7"}),
        ("signal", "signal_key", "cluster-b/node-a/gpu_inventory_mismatch/node"),
        ("signal", "active", None),
        ("signal", "active", "false"),
        ("signal", "observed_at", (NOW - timedelta(seconds=331)).isoformat()),
        ("signal", "clock_at", None),
    ],
)
def test_inventory_rejects_missing_mixed_or_stale_evidence(
    target: str, key: str, value: Any
) -> None:
    proof = inventory_proof()
    row = {
        "root": proof,
        "status": proof["statuses"][0],
        "metric": proof["metrics"][0],
        "signal": proof["signals"]["gpu_inventory_mismatch"],
    }[target]
    row[key] = value
    assert inventory_errors(proof), f"accepted {target}.{key}={value!r}"


def test_duplicate_inventory_metric_cannot_mask_a_disagreeing_source() -> None:
    proof = inventory_proof()
    proof["metrics"].append(deepcopy(proof["metrics"][0]))
    assert inventory_errors(proof), (
        "duplicate inventory metrics must not count as unique source evidence"
    )


def test_never_active_signal_is_not_required_to_create_a_store_row() -> None:
    proof = inventory_proof()
    proof["signals"]["gpu_inventory_mismatch"] = None
    assert inventory_errors(proof) == []
    proof["metrics"][0]["labels"]["consecutive_mismatch_samples"] = "1"
    assert inventory_errors(proof), "absence cannot replace healthy sample evidence"


@pytest.mark.parametrize(
    "problem", [None, "gpu", "efa", "runtime", "digest", "partial", "zero-efa"]
)
def test_collect003_invokes_scoped_probe_and_running_configuration(
    problem: str | None,
) -> None:
    env = dict(ENV)
    proof = inventory_proof()
    if problem == "zero-efa":
        env["GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT"] = "0"
        for row in proof["metrics"]:
            if row["name"].startswith("efa_"):
                row["value"] = 0
                row["labels"].update(expected_count="0", observed_count="0")
    file = {"sha256": "a" * 64}
    runtime = {
        "invocation_id": "invocation-a",
        "file_env": dict(env),
        "running_env": dict(env),
        "file": file,
    }
    if problem == "runtime":
        runtime["running_env"]["GPU_FAULT_EXPECTED_GPU_COUNT"] = "7"
    elif problem == "digest":
        runtime["file"] = {"sha256": "b" * 64}
    elif problem == "partial":
        runtime["file_env"].pop("GPU_FAULT_EXPECTED_GPU_COUNT")
    calls = []
    fixture = SimpleNamespace(
        node="node-a",
        regional=SimpleNamespace(
            settings=SimpleNamespace(cluster_id="cluster-a"),
            cpu_python=lambda *args: calls.append(args) or proof,
        ),
        execute=lambda command: (calls.append(command) or runtime),
        snapshot=lambda: {
            "collector_env": env,
            "collector_env_file": file,
            "gpu_inventory": [{}] * (7 if problem == "gpu" else 8),
            "efa_inventory": {
                "active_count": 0
                if problem == "zero-efa"
                else 3
                if problem == "efa"
                else 4
            },
        },
    )
    result = runner.run_collect003(fixture)
    assert result["verdict"] == ("PASS" if problem in {None, "zero-efa"} else "FAIL")
    assert calls == [
        "inventory-config",
        (evidence.INVENTORY_STATE_PROBE, "cluster-a", "node-a"),
    ]
    assert result["control_plane_inventory"]["signals"]


@pytest.mark.parametrize("value", ["invalid", "-1", None, True])
def test_collect003_rejects_unknown_or_negative_expected_efa(value: Any) -> None:
    fixture = SimpleNamespace(
        snapshot=lambda: {
            "collector_env": {**ENV, "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT": value},
            "gpu_inventory": [{}] * 8,
            "efa_inventory": {"active_count": 4},
        }
    )
    with pytest.raises(runner.RegionalFixtureError, match="nonnegative"):
        runner.run_collect003(fixture)


def mismatch_record(
    count: int,
    *,
    seconds: int = 0,
    value: int = 0,
    required: int = 2,
    expected: int = 9,
) -> dict[str, Any]:
    return {
        "record_id": f"host-telemetry/host-node-a-{seconds}",
        "batch_id": f"host-node-a-{seconds}",
        "observed_at": (NOW + timedelta(seconds=seconds)).isoformat(),
        "edge_filter_reasons": ["baseline"] if count == 1 else ["threshold"],
        "samples": [
            {
                "name": "gpu_inventory_mismatch",
                "value": value,
                "labels": {
                    "consecutive_mismatch_samples": str(count),
                    "required_consecutive_samples": str(required),
                    "expected_count": str(expected),
                    "observed_count": str(expected - 1),
                },
            }
        ],
    }


def debounce(records: list[dict[str, Any]], **kwargs: Any) -> list[str]:
    return evidence.debounce_sample_errors(
        records,
        **{
            "interval": 15,
            "required_samples": 2,
            "tolerance": 0.5,
            "expected_count": 9,
            **kwargs,
        },
    )


def test_debounce_is_first_to_threshold_and_ignores_runner_clock() -> None:
    assert debounce([]), "no samples cannot prove debounce"
    records = [mismatch_record(1), mismatch_record(2, seconds=15, value=1)]
    assert debounce(records) == []
    for skew in (-3600, 3600):
        assert (
            destructive.debounce_errors(
                records,
                interval=15,
                required_samples=2,
                tolerance=0.5,
                started_at=NOW + timedelta(seconds=skew),
                expected_count=9,
            )
            == []
        )
    assert debounce([records[1]]), "a threshold label is not first-sample evidence"
    assert debounce([mismatch_record(1, expected=8), *records]) == [], (
        "an older configuration is not this run's debounce episode"
    )
    assert debounce([mismatch_record(0), *records]) == [], (
        "a healthy sample is not the first mismatch"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval": 0},
        {"interval": True},
        {"required_samples": 1},
        {"required_samples": True},
        {"tolerance": float("nan")},
        {"tolerance": -1},
        {"tolerance": 2},
    ],
)
def test_debounce_invalid_configuration_is_not_a_looser_verdict(
    kwargs: dict[str, Any],
) -> None:
    assert debounce([], **kwargs), (
        "invalid debounce configuration must fail closed even without samples"
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing-first",
        "missing-threshold",
        "first-fired",
        "threshold-late",
        "threshold-early",
        "wrong-required",
        "missing-label",
        "bad-label",
        "not-one-missing",
        "bad-time",
        "bad-value",
        "no-baseline",
        "regressed",
        "duplicate",
        "count-too-late",
    ],
)
def test_debounce_rejects_missing_or_inconsistent_collector_samples(
    change: str,
) -> None:
    first = mismatch_record(1)
    threshold = mismatch_record(2, seconds=15, value=1)
    records = [first, threshold]
    labels = threshold["samples"][0]["labels"]
    if change == "missing-first":
        records.pop(0)
    elif change == "missing-threshold":
        records.pop()
    elif change == "first-fired":
        first["samples"][0]["value"] = 1
    elif change in {"threshold-late", "threshold-early"}:
        threshold["observed_at"] = (
            NOW + timedelta(seconds=60 if change == "threshold-late" else 1)
        ).isoformat()
    elif change == "wrong-required":
        labels["required_consecutive_samples"] = "3"
    elif change == "missing-label":
        labels.pop("expected_count")
    elif change == "bad-label":
        labels["expected_count"] = "not-a-count"
    elif change == "not-one-missing":
        labels["observed_count"] = "7"
    elif change == "bad-time":
        first["observed_at"] = "2026-09-15T00:00:00"
    elif change == "bad-value":
        first["samples"][0]["value"] = float("nan")
    elif change == "no-baseline":
        first["edge_filter_reasons"] = ["health-summary"]
    elif change == "regressed":
        records.insert(1, mismatch_record(1, seconds=5))
    elif change == "duplicate":
        records.append(deepcopy(threshold))
    elif change == "count-too-late":
        labels["consecutive_mismatch_samples"] = "3"
    assert debounce(records), f"accepted {change}"


def test_actual_host_producer_supplies_baseline_and_threshold_within_one_interval(
    tmp_path: Path,
) -> None:
    batches = []
    stamps = iter([NOW, NOW + timedelta(seconds=15)])
    sink = SimpleNamespace(post=lambda _path, payload: batches.append(payload) or {})
    collector = HostTelemetryCollector(
        sink,
        CollectorContext(cluster_id="cluster-a"),
        node_id="node-a",
        expected_gpu_count=9,
        expected_efa_device_count=4,
        proc_root=str(tmp_path),
        infiniband_root=str(tmp_path),
        now=lambda: next(stamps),
        edge_filter_enabled=True,
        runner=lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, "\n".join(f"GPU-{index}, 0" for index in range(8)), ""
        ),
    )
    collector.CONTRIBUTORS = ("_gpu_inventory",)
    for _ in range(2):
        collector.collect_once()
    records = [
        {
            "observed_at": batch["observed_at"],
            "edge_filter_reasons": batch["edge_filter_reasons"],
            "samples": batch["samples"],
        }
        for batch in batches
    ]
    assert len(batches) == 2
    assert debounce(records) == []
    assert records[0]["samples"][-1]["value"] == 0
    assert records[1]["samples"][-1]["value"] == 1


def test_control_plane_probe_reads_actual_scoped_state_and_keeps_recovered_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch

    store = InMemoryStore()
    now = datetime.now(timezone.utc)
    samples = []
    for resource, count in (("gpu", 8), ("efa", 4)):
        labels = {
            "expected_count": str(count),
            "observed_count": str(count),
            "consecutive_mismatch_samples": "0",
            "missing_count": "0",
            "excess_count": "0",
        }
        samples.extend(
            [
                HostMetricSample(
                    name=f"{resource}_inventory_expected_count",
                    value=count,
                    labels=labels,
                ),
                HostMetricSample(
                    name=f"{resource}_inventory_active_count",
                    value=count,
                    labels=labels,
                ),
                HostMetricSample(
                    name=f"{resource}_inventory_mismatch", value=0, labels=labels
                ),
            ]
        )
    policy = NodeHealthPolicy(store)
    policy.evaluate_metrics(
        HostTelemetryBatch(
            batch_id="host-node-a-1",
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=now,
            received_at=now,
            samples=samples,
        )
    )
    store.save_collector_status(
        CollectorStatus(
            cluster_id="cluster-a",
            node_id="node-a",
            collector=CollectorKind.HOST_TELEMETRY,
            observed_at=now,
            ingested_at=now,
            last_success_at=now,
            batch_id="host-node-a-1",
            sample_count=len(samples),
        )
    )
    monkeypatch.setattr(
        ApplicationContext, "from_environment", lambda: SimpleNamespace(store=store)
    )
    monkeypatch.setattr(sys, "argv", ["probe", "cluster-a", "node-a"])
    output = io.StringIO()
    with redirect_stdout(output):
        exec(
            compile(evidence.INVENTORY_STATE_PROBE, "<inventory-state-probe>", "exec"),
            {},
        )
    proof = json.loads(output.getvalue())
    assert inventory_errors(proof) == []
    assert set(proof["signals"]) == {"gpu_inventory_mismatch", "efa_inventory_mismatch"}


def prepare_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, str]]:
    process = tmp_path / "123"
    process.mkdir()
    (process / "stat").write_text("123 (host collector) " + " ".join(["0"] * 20))
    (process / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in ENV.items())
        + b"\0GPU_FAULT_CONTROL_PLANE_TOKEN=unit-only-private-value\0"
    )
    unit = {"ActiveState": "active", "MainPID": "123", "InvocationID": "invocation-a"}
    config = tmp_path / "collector.env"
    config.write_text("\n".join(f"{key}={value}" for key, value in ENV.items()))
    monkeypatch.setattr(probe, "COLLECTOR_ENV", config)
    monkeypatch.setattr(probe, "PROC_ROOT", tmp_path)
    monkeypatch.setattr(
        probe, "service_snapshot", lambda: {probe.HOST_COLLECTOR_UNIT: unit}
    )
    return process, unit


def test_inventory_runtime_probe_projects_only_bound_numeric_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prepare_process(tmp_path, monkeypatch)
    arguments = probe.parser().parse_args(["inventory-config"])
    arguments.handler(arguments)
    text = capsys.readouterr().out
    proof = json.loads(text)
    assert proof["running_env"] == ENV
    assert proof["file_env"] == ENV
    assert len(proof["file"]["sha256"]) == 64
    assert "unit-only-private-value" not in text


@pytest.mark.parametrize(
    "problem",
    [
        "inactive",
        "pid",
        "pid-one",
        "invocation",
        "restart",
        "reused-pid",
        "config",
        "numeric",
    ],
)
def test_inventory_runtime_read_refuses_unknown_or_changing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    process, unit = prepare_process(tmp_path, monkeypatch)
    if problem == "inactive":
        unit["ActiveState"] = "inactive"
    elif problem == "pid":
        unit["MainPID"] = ""
    elif problem == "pid-one":
        unit["MainPID"] = "1"
    elif problem == "invocation":
        unit["InvocationID"] = ""
    elif problem == "restart":
        units = iter([unit, {**unit, "InvocationID": "replacement"}])
        monkeypatch.setattr(
            probe, "service_snapshot", lambda: {probe.HOST_COLLECTOR_UNIT: next(units)}
        )
    elif problem == "reused-pid":
        calls = 0

        def states() -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 2:
                (process / "stat").write_text("123 (host) " + " ".join(["1"] * 20))
            return {probe.HOST_COLLECTOR_UNIT: unit}

        monkeypatch.setattr(probe, "service_snapshot", states)
    elif problem == "config":
        readings = iter([{"sha256": "before"}, {"sha256": "after"}])
        monkeypatch.setattr(probe, "file_snapshot", lambda _path: next(readings))
    elif problem == "numeric":
        (process / "environ").write_bytes(b"GPU_FAULT_EXPECTED_GPU_COUNT=secret\0")
    with pytest.raises(probe.ProbeError):
        probe.inventory_config(SimpleNamespace())
