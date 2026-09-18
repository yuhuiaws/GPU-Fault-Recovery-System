"""Evaluate revised dashboard queries with the same real PromQL engine as AMP."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = lazy_script_module(ROOT / "scripts/build-grafana-dashboards.py")
LABELS = {"control_plane_cluster": "control-a", "region": "test-region"}
PREFIX = "gpu_fault_processor_counter_"


def panel_expression(title: str, target: int = 0) -> str:
    panels = [
        panel
        for dashboard in GENERATOR.DASHBOARDS
        for row in dashboard.rows
        for panel in row.panels
        if panel.title == title
    ]
    assert len(panels) == 1, "the dashboard query must come from one actual panel"
    return (
        panels[0]
        .targets[target]
        .expr.replace("$control_plane_cluster", LABELS["control_plane_cluster"])
        .replace("$region", LABELS["region"])
    )


def sample(name: str, values: str, *, pod="worker-a", process="0", **extra) -> dict:
    labels = {
        **LABELS,
        "pod": pod,
        "service_role": "gpu-fault-control-worker",
        "instance": f"{pod}:8000",
        "job": "gpu-fault-control-plane",
        **extra,
    }
    if process is not None:
        labels["process"] = process
    return {
        "series": name
        + "{"
        + ",".join(f"{key}={json.dumps(value)}" for key, value in labels.items())
        + "}",
        "values": values,
    }


def expected(value: float | None, **labels) -> list[dict]:
    if value is None:
        return []
    return [
        {
            "labels": "{"
            + ",".join(
                f"{key}={json.dumps(item)}"
                for key, item in {**LABELS, **labels}.items()
            )
            + "}",
            "value": value,
        }
    ]


@pytest.fixture(scope="module")
def promtool() -> str:
    path = shutil.which(os.getenv("PROMTOOL", "promtool"))
    if path is None:
        pytest.fail("the pinned promtool is required for dashboard behavior tests")
    return path


def evaluate(tmp_path: Path, promtool: str, inputs: list[dict], queries: list[dict]):
    path = tmp_path / "dashboard-test.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "evaluation_interval": "1m",
                "tests": [
                    {
                        "interval": "1m",
                        "input_series": inputs,
                        "promql_expr_test": [
                            {"eval_time": "20m", **query} for query in queries
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [promtool, "test", "rules", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "condition", ["fresh", "repaired", "stale", "future", "partial"]
)
def test_drift_dashboard_uses_only_the_newest_complete_current_scan(
    tmp_path, promtool, condition
) -> None:
    stamp = (
        "3600+60x30"
        if condition == "future"
        else ("600+0x30" if condition == "stale" else "0+60x30")
    )
    inputs = [
        sample("up", "1+0x30", process=None),
        sample(PREFIX + "drift_abs", "7+0x30"),
        sample(PREFIX + "drift_scan_timestamp_seconds", stamp),
        sample(PREFIX + "drift_scan_max_age_seconds", "180+0x30"),
    ]
    if condition != "partial":
        inputs.append(sample(PREFIX + "mismatched_clusters", "2+0x30"))
    if condition == "repaired":
        inputs[2] = sample(PREFIX + "drift_scan_timestamp_seconds", "1080+0x30")
        inputs.extend(
            sample(PREFIX + suffix, values, pod="worker-new", process="1")
            for suffix, values in (
                ("drift_abs", "0+0x30"),
                ("mismatched_clusters", "0+0x30"),
                ("drift_scan_timestamp_seconds", "0+60x30"),
                ("drift_scan_max_age_seconds", "180+0x30"),
            )
        )
    known = condition in {"fresh", "repaired"}
    evaluate(
        tmp_path,
        promtool,
        inputs,
        [
            {
                "expr": panel_expression("Processor counter drift", target),
                "exp_samples": expected(
                    (0 if condition == "repaired" else value) if known else None
                ),
            }
            for target, value in enumerate((7, 2))
        ]
        + [
            {
                "expr": panel_expression("Processor counter scan unavailable"),
                "exp_samples": expected(0 if known else 1),
            }
        ],
    )


@pytest.mark.parametrize("unavailable", ["stale", "partial"])
def test_unknown_current_scan_does_not_retain_a_historical_zero(
    tmp_path, promtool, unavailable
):
    inputs = [
        sample("up", "1+0x30", process=None),
        sample(PREFIX + "drift_abs", "0+0x30"),
        sample(
            PREFIX + "mismatched_clusters",
            "0+0x18 -1+0x11" if unavailable == "partial" else "0+0x30",
        ),
        sample(
            PREFIX + "drift_scan_timestamp_seconds",
            "0+60x16 960+0x13" if unavailable == "stale" else "0+60x30",
        ),
        sample(PREFIX + "drift_scan_max_age_seconds", "180+0x30"),
    ]
    evaluate(
        tmp_path,
        promtool,
        inputs,
        [
            {
                "expr": panel_expression("Processor counter drift", target),
                "exp_samples": [],
            }
            for target in (0, 1)
        ]
        + [
            {
                "expr": panel_expression("Processor counter scan unavailable"),
                "exp_samples": expected(1),
            }
        ],
    )


@pytest.mark.parametrize(("last_seen", "active"), [(600, 1), (60, 0), (0, 0)])
def test_terminal_notification_panel_counts_events_not_retained_population(
    tmp_path, promtool, last_seen, active
) -> None:
    evaluate(
        tmp_path,
        promtool,
        [
            sample("gpu_fault_notification_total", "1+0x30", status="FAILED"),
            sample(
                "gpu_fault_notification_terminal_failure_last_seen_timestamp_seconds",
                f"{last_seen}+0x30",
            ),
        ],
        [
            {
                "expr": panel_expression("Terminal notification failure seen (15m)"),
                "exp_samples": expected(active),
            }
        ],
    )


@pytest.mark.parametrize("verdict", [None, 0, 1])
def test_process_coverage_panel_keeps_missing_verdict_unknown(
    tmp_path, promtool, verdict
):
    inputs = [sample("up", "1+0x30", process=None)]
    if verdict is not None:
        inputs.append(
            sample(
                "gpu_fault_metrics_aggregation_degraded",
                f"{verdict}+0x30",
                process=None,
            )
        )
    evaluate(
        tmp_path,
        promtool,
        inputs,
        [
            {
                "expr": panel_expression("Process metric coverage incomplete"),
                "exp_samples": expected(
                    0 if verdict == 0 else 1,
                    pod="worker-a",
                    service_role="gpu-fault-control-worker",
                ),
            }
        ],
    )


def test_periodic_job_chart_does_not_group_by_the_prometheus_scrape_job(
    tmp_path, promtool
):
    metric = "gpu_fault_periodic_job_error_last_seen_timestamp_seconds"
    evaluate(
        tmp_path,
        promtool,
        [
            sample(metric, "600+0x30", periodic_job="cleanup"),
            sample(metric, "660+0x30", periodic_job="counter_drift"),
        ],
        [
            {
                "expr": panel_expression("Periodic service error last seen age", 1),
                "exp_samples": [
                    *expected(600, periodic_job="cleanup"),
                    *expected(540, periodic_job="counter_drift"),
                ],
            }
        ],
    )


def test_aurora_panel_does_not_hide_a_failed_projection_with_a_healthy_replica(
    tmp_path, promtool
):
    metric = "gpu_fault_aurora_credential_refresh_last_run_ok"
    evaluate(
        tmp_path,
        promtool,
        [sample(metric, "0+0x30"), sample(metric, "1+0x30", pod="worker-b")],
        [
            {
                "expr": panel_expression("Aurora credential refresh", 1),
                "exp_samples": expected(0),
            }
        ],
    )
