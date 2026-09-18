"""Evaluate the shipped O1 rules with Prometheus, without an external service."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from gpu_fault.app.closed_loop_metrics import closed_loop_window_metric_lines
from gpu_fault.app.metric_scan_cache import WorkflowScan
from gpu_fault.app.process_metrics import parse_lines
from tests.metrics.test_closed_loop_metrics import (
    COMPLETE,
    COUNT,
    END,
    FAMILIES,
    MEAN,
    NOW,
    fresh_health_dag,
    scan,
    workflow,
)

ROOT = Path(__file__).resolve().parents[2]
SLOW = "GpuFaultClosedLoopSlow"
UNKNOWN = "GpuFaultClosedLoopWindowIncomplete"
LABELS = {"control_plane_cluster": "control-a", "region": "test-region"}
WORKER_ROLE = "gpu-fault-control-worker"


def rules(path: str, *, wrapped: bool = False) -> dict[str, dict]:
    document = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    groups = document["spec"]["groups"] if wrapped else document["groups"]
    return {
        rule["alert"]: rule
        for group in groups
        for rule in group["rules"]
        if rule.get("alert") in {SLOW, UNKNOWN}
    }


def amp_rules() -> dict[str, dict]:
    return rules("deploy/observability/amp-rules.yaml")


def test_operator_and_amp_rules_are_equivalent() -> None:
    operator = rules(
        "deploy/control-plane/regional/processor-alerts.yaml", wrapped=True
    )
    amp = amp_rules()

    assert set(amp) == {SLOW, UNKNOWN}
    assert amp == operator


def test_all_window_metrics_survive_the_control_plane_adot_keep_filter() -> None:
    documents = yaml.safe_load_all(
        (ROOT / "deploy/observability/adot-control-plane.yaml").read_text(
            encoding="utf-8"
        )
    )
    config_map = next(item for item in documents if item["kind"] == "ConfigMap")
    config = yaml.safe_load(config_map["data"]["collector.yaml"])
    jobs = config["receivers"]["prometheus"]["config"]["scrape_configs"]
    job = next(item for item in jobs if item["job_name"] == "gpu-fault-control-plane")
    relabels = job["relabel_configs"]
    role = next(item for item in relabels if item.get("target_label") == "service_role")
    app_keep = next(
        item
        for item in relabels
        if item.get("action") == "keep"
        and item.get("source_labels") == ["__meta_kubernetes_pod_label_app"]
    )
    assert role["source_labels"] == ["__meta_kubernetes_pod_label_app"], (
        "PromQL service_role fixtures must use ADOT's app-label value"
    )
    assert re.fullmatch(app_keep["regex"], WORKER_ROLE), (
        "the fixture must model an actual scraped control-worker, not an invented role"
    )
    keeps = [
        item["regex"]
        for item in job["metric_relabel_configs"]
        if item["action"] == "keep" and item["source_labels"] == ["__name__"]
    ]
    emitted = closed_loop_window_metric_lines(
        WorkflowScan((), limit=20, truncated=False, observed_at=NOW)
    )
    names = {line.split("{", 1)[0].split()[0] for line in emitted if line[0] != "#"}

    assert names == FAMILIES
    assert keeps, "ADOT must retain an explicit metric-name allowlist"
    assert all(any(re.fullmatch(keep, name) for keep in keeps) for name in names), (
        "every emitted window family must survive the production ADOT keep filter"
    )


def series(
    name: str,
    values: str,
    *,
    pod: str = "worker-a",
    milestone: bool = False,
    service_role: str = WORKER_ROLE,
) -> dict[str, str]:
    labels = {
        **LABELS,
        "job": "gpu-fault-control-plane",
        "namespace": "control",
        "service_role": service_role,
        "pod": pod,
        "instance": f"{pod}:8000",
    }
    if milestone:
        labels["milestone"] = "containment"
    rendered = ",".join(f"{key}={json.dumps(value)}" for key, value in labels.items())
    return {"series": f"{name}{{{rendered}}}", "values": values}


def replica(
    *,
    pod: str = "worker-a",
    mean: str = "600",
    count: str = "2",
    complete: str = "1",
    end: str = "0+60x30",
    missing: frozenset[str] = frozenset(),
    starts_at: int = 0,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, values, milestone in (
        ("up", "1+0x30", False),
        ("gpu_fault_workflow_scan_limit", "20000+0x30", False),
        (MEAN, " ".join([mean] * 31), True),
        (COUNT, " ".join([count] * 31), True),
        (COMPLETE, " ".join([complete] * 31), True),
        (END, end, False),
    ):
        if name in missing:
            continue
        if starts_at:
            if name == END:
                values = f"{starts_at * 60}+60x{30 - starts_at}"
            elif name in (MEAN, COUNT, COMPLETE):
                values = " ".join(values.split()[starts_at:])
            else:
                constant = "1" if name == "up" else "20000"
                values = f"{constant}+0x{30 - starts_at}"
            values = " ".join(["_"] * starts_at) + " " + values
        rows.append(series(name, values, pod=pod, milestone=milestone))
    return rows


def window_replica(scenario: str) -> list[dict[str, str]]:
    if scenario == "initial-dag":
        row = fresh_health_dag()
        observed = scan(row, at=row.created_at)
    else:
        assert scenario == "concurrent-completion", (
            "only explicit window regressions are supported"
        )
        observed = scan(workflow(at=NOW + timedelta(seconds=1), seconds=600))
    rendered = parse_lines(closed_loop_window_metric_lines(observed))
    values = {
        family: next(
            sample.value
            for sample in rendered.samples[family]
            if dict(sample.labels).get("milestone") == "containment"
        )
        for family in (MEAN, COUNT, COMPLETE)
    }
    return replica(mean=values[MEAN], count=values[COUNT], complete=values[COMPLETE])


@pytest.fixture(scope="module")
def promtool() -> str:
    candidate = os.environ.get("PROMTOOL", "promtool")
    resolved = shutil.which(candidate)
    if resolved is None:
        pytest.fail("promtool is required; set PROMTOOL to the pinned local binary")
    return resolved


@pytest.mark.parametrize(
    ("inputs", "slow_value", "unknown"),
    [
        pytest.param(
            "initial-dag", None, False, id="fresh-health-dag-has-complete-window"
        ),
        pytest.param(
            "concurrent-completion",
            None,
            False,
            id="post-boundary-completion-is-known-outside-window",
        ),
        pytest.param(replica(), None, False, id="fast"),
        pytest.param(replica(mean="1200"), 1200, False, id="slow"),
        pytest.param(replica(mean="900"), None, False, id="at-threshold"),
        pytest.param(
            replica(mean="NaN", count="0"), None, False, id="empty-is-not-unknown"
        ),
        pytest.param(
            replica(mean="NaN", count="NaN", complete="0"), None, True, id="truncated"
        ),
        pytest.param(
            replica(mean="1200", complete="0"),
            None,
            True,
            id="partial-mean-cannot-fire-slow",
        ),
        pytest.param(
            replica() + replica(pod="worker-b", mean="NaN", count="NaN", complete="0"),
            None,
            True,
            id="healthy-replica-cannot-hide-truncation",
        ),
        pytest.param(
            replica(mean="100", count="100")
            + replica(pod="worker-b", mean="1200", count="1"),
            1200,
            False,
            id="mean-is-paired-before-replica-aggregation",
        ),
        pytest.param(
            replica() + replica(pod="worker-new", starts_at=10),
            None,
            False,
            id="restart-does-not-create-observations",
        ),
        pytest.param(replica(mean="1200", end="0+0x30"), None, True, id="stale-scan"),
        pytest.param(
            replica(mean="1200", end="3600+60x30"), None, True, id="future-scan"
        ),
        pytest.param(replica(missing=frozenset({MEAN})), None, True, id="missing-mean"),
        pytest.param(
            replica(missing=frozenset({COUNT})), None, True, id="missing-count"
        ),
        pytest.param(
            replica(missing=frozenset({COMPLETE})), None, True, id="missing-coverage"
        ),
        pytest.param(
            replica(missing=frozenset({END})), None, True, id="missing-scan-time"
        ),
        pytest.param(
            replica(missing=frozenset(FAMILIES)),
            None,
            True,
            id="old-worker-without-window-metrics",
        ),
        pytest.param(
            replica(missing=frozenset(FAMILIES | {"gpu_fault_workflow_scan_limit"})),
            None,
            True,
            id="failed-contributor-still-has-up",
        ),
        pytest.param(
            replica()
            + replica(
                pod="worker-b",
                missing=frozenset(FAMILIES | {"gpu_fault_workflow_scan_limit"}),
            ),
            None,
            True,
            id="other-replica-cannot-hide-failed-contributor",
        ),
        pytest.param(
            [series("up", "1+0x30", service_role="gpu-fault-api-ha")],
            None,
            False,
            id="ingress-up-is-not-a-missing-worker",
        ),
        pytest.param(
            [series("up", "1+0x30", service_role="gpu-fault-telemetry-spool-worker")],
            None,
            False,
            id="spool-up-is-not-a-missing-worker",
        ),
        pytest.param(
            replica()
            + [
                series(
                    "gpu_fault_closed_loop_milestone_seconds_sum",
                    "200+0x4 1200+0x25",
                    milestone=True,
                ),
                series(
                    "gpu_fault_closed_loop_milestone_seconds_count",
                    "2+0x30",
                    milestone=True,
                ),
            ],
            None,
            False,
            id="snapshot-turnover-is-not-a-1000-second-event",
        ),
    ],
)
def test_closed_loop_promql(
    tmp_path: Path,
    promtool: str,
    inputs: list[dict[str, str]] | str,
    slow_value: int | None,
    unknown: bool,
) -> None:
    if isinstance(inputs, str):
        inputs = window_replica(inputs)
    selected = amp_rules()
    rule_path = tmp_path / "rules.yaml"
    rule_path.write_text(
        yaml.safe_dump(
            {"groups": [{"name": "closed-loop", "rules": list(selected.values())}]}
        ),
        encoding="utf-8",
    )
    alerts: list[dict] = []
    queries: list[dict] = []
    for name, value in ((SLOW, slow_value), (UNKNOWN, 20000 if unknown else None)):
        # A contributor that emits nothing has only up=1 as its absence anchor.
        if (
            name == UNKNOWN
            and unknown
            and not any(
                item["series"].startswith("gpu_fault_workflow_scan_limit{")
                and 'pod="worker-a"' in item["series"]
                for item in inputs
            )
        ):
            value = 1
        elif (
            name == UNKNOWN
            and unknown
            and any('pod="worker-b"' in item["series"] for item in inputs)
        ):
            absent_b = not any(
                item["series"].startswith("gpu_fault_workflow_scan_limit{")
                and 'pod="worker-b"' in item["series"]
                for item in inputs
            )
            if absent_b:
                value = 1
        labels = ",".join(f"{key}={json.dumps(val)}" for key, val in LABELS.items())
        queries.append(
            {
                "expr": selected[name]["expr"],
                "eval_time": "20m",
                "exp_samples": (
                    []
                    if value is None
                    else [{"labels": "{" + labels + "}", "value": value}]
                ),
            }
        )
        alerts.append(
            {
                "alertname": name,
                "eval_time": "20m",
                "exp_alerts": (
                    []
                    if value is None
                    else [
                        {
                            "exp_labels": {**LABELS, "severity": "warning"},
                            "exp_annotations": selected[name]["annotations"],
                        }
                    ]
                ),
            }
        )
    test_path = tmp_path / "test.yaml"
    test_path.write_text(
        yaml.safe_dump(
            {
                "rule_files": [str(rule_path)],
                "evaluation_interval": "1m",
                "tests": [
                    {
                        "interval": "1m",
                        "input_series": inputs,
                        "promql_expr_test": queries,
                        "alert_rule_test": alerts,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [promtool, "test", "rules", str(test_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
