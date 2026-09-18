"""Real PromQL checks for reset domains and current diagnostic evidence."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from gpu_fault.app.metric_aggregation import STRATEGIES, Strategy
from gpu_fault.app.process_metrics import aggregate, parse_lines

ROOT = Path(__file__).resolve().parents[2]
BASE = {
    "control_plane_cluster": "control-a",
    "region": "test-region",
    "job": "gpu-fault-control-plane",
    "service_role": "gpu-fault-control-worker",
    "namespace": "control",
}
DRIFT = "gpu_fault_processor_counter_drift_abs"
MISMATCH = "gpu_fault_processor_counter_mismatched_clusters"
SCAN = "gpu_fault_processor_counter_drift_scan_timestamp_seconds"
MAX_AGE = "gpu_fault_processor_counter_drift_scan_max_age_seconds"
TERMINAL = "gpu_fault_notification_terminal_failure_last_seen_timestamp_seconds"
REFRESH = "gpu_fault_aurora_credential_refresh_"
COVERAGE = "GpuFaultMetricsAggregationIncomplete"
DRIFT_ALERT = "GpuFaultProcessorCounterDrift"
SCAN_UNKNOWN = "GpuFaultProcessorCounterDriftScanUnavailable"
NOTIFY_ALERT = "GpuFaultNotificationDeliveryFailing"
REFRESH_UNKNOWN = "GpuFaultAuroraCredentialRefreshStatusUnknown"
REFRESH_FAILED = "GpuFaultAuroraCredentialRefreshFailing"
REFRESH_STALE = "GpuFaultAuroraCredentialRefreshStale"
SWEEP = "GpuFaultRemoteCommandStaleFenceSwept"
PERIODIC_ERROR = "GpuFaultPeriodicServiceErrors"
MIRRORED = {
    COVERAGE,
    DRIFT_ALERT,
    SCAN_UNKNOWN,
    NOTIFY_ALERT,
    REFRESH_UNKNOWN,
    REFRESH_FAILED,
    REFRESH_STALE,
    SWEEP,
    PERIODIC_ERROR,
}
COUNTER_ALERTS = {
    "gpu_fault_workflow_lifetime_exceeded_total": "GpuFaultWorkflowLifetimeExceeded",
    "gpu_fault_processor_admission_rejections_total": "GpuFaultProcessorAdmissionRejected",
}


def rule_map(relative: str, *, wrapped: bool = False):
    document = yaml.safe_load((ROOT / relative).read_text())
    groups = document["spec"]["groups"] if wrapped else document["groups"]
    return {rule["alert"]: rule for group in groups for rule in group["rules"]}


def series(
    name: str,
    values: str,
    *,
    pod: str = "worker-a",
    process: str | None = None,
    **labels: str,
):
    tags = {**BASE, "pod": pod, "instance": f"{pod}:8000", **labels}
    if process is not None:
        tags["process"] = process
    rendered = ",".join(f"{key}={json.dumps(value)}" for key, value in tags.items())
    return {"series": f"{name}{{{rendered}}}", "values": values}


def scan(
    *,
    pod: str = "worker-a",
    process: str = "0",
    drift: str = "0+0x60",
    mismatched: str = "0+0x60",
    at: str = "0+60x60",
    age: str = "180+0x60",
    missing: frozenset[str] = frozenset(),
):
    return [
        series(name, values, pod=pod, process=process)
        for name, values in (
            (DRIFT, drift),
            (MISMATCH, mismatched),
            (SCAN, at),
            (MAX_AGE, age),
        )
        if name not in missing
    ]


def expected(alert: str, value: int, when: str = "40m"):
    return {
        "expr": f'count(ALERTS{{alertname="{alert}",alertstate="firing"}}) or vector(0)',
        "eval_time": when,
        "exp_samples": [{"labels": "{}", "value": value}],
    }


def counter_cases():
    result = []
    for metric, alert in COUNTER_ALERTS.items():
        for label, after, fired in (
            ("stable", (7, 11), 0),
            ("partial-restart", (0, 11), 0),
            ("whole-restart", (0, 0), 0),
            ("survivor-increment", (0, 21), 1),
        ):

            def render(values):
                peers = []
                for slot, value in enumerate(values):
                    peer = parse_lines(
                        [f"# TYPE {metric} counter", f"{metric} {value}"]
                    )
                    peer.slot = slot
                    peers.append(peer)
                return parse_lines(aggregate(peers[0], peers[1:])).samples[metric]

            before, after_samples = render((7, 11)), render(after)
            assert [dict(item.labels) for item in before] == [
                {"process": "0"},
                {"process": "1"},
            ]
            inputs = [
                series(
                    metric,
                    f"{old.value}+0x30 {new.value}+0x30",
                    process=dict(old.labels)["process"],
                )
                for old, new in zip(before, after_samples, strict=True)
            ]
            result.append(
                pytest.param(
                    inputs, [expected(alert, fired, "33m")], id=metric + "-" + label
                )
            )
    return result


def cases():
    result = counter_cases()

    def add(name, inputs, checks):
        result.append(pytest.param(inputs, checks, id=name))

    up = series("up", "1+0x60")
    for label, degraded, fired in (
        ("complete", "0+0x60", 0),
        ("degraded", "1+0x60", 1),
        ("missing-coverage", None, 1),
    ):
        inputs = [up]
        if degraded is not None:
            inputs.append(series("gpu_fault_metrics_aggregation_degraded", degraded))
        add(label, inputs, [expected(COVERAGE, fired, "7m")])
    add(
        "coverage-recovered",
        [up, series("gpu_fault_metrics_aggregation_degraded", "1+0x9 0+0x50")],
        [expected(COVERAGE, 1, "7m"), expected(COVERAGE, 0, "11m")],
    )

    for label, inputs, drift, unknown in (
        ("aligned", scan(), 0, 0),
        ("real-drift", scan(drift="7+0x60"), 1, 0),
        ("cluster-only-mismatch", scan(mismatched="1+0x60"), 1, 0),
        ("never-scanned", scan(at="0+0x60"), 0, 1),
        ("expired-scan", scan(drift="7+0x60", at="600+0x60"), 0, 1),
        ("future-scan", scan(drift="7+0x60", at="3600+60x60"), 0, 1),
        ("invalid-age-limit", scan(age="0+0x60"), 0, 1),
        ("nonfinite-scan", scan(at="NaN+0x60"), 0, 1),
        ("missing-values", scan(missing=frozenset({MISMATCH})), 0, 1),
        ("missing-time", scan(missing=frozenset({SCAN})), 0, 1),
        ("missing-age", scan(missing=frozenset({MAX_AGE})), 0, 1),
    ):
        add(
            label,
            [up, *inputs],
            [expected(DRIFT_ALERT, drift), expected(SCAN_UNKNOWN, unknown)],
        )
    for second_pod, second_slot in (("worker-a", "1"), ("worker-b", "0")):
        for current, fired in (("0", 0), ("7", 1)):
            inputs = [
                up,
                *scan(drift="7+0x60", at="0+60x19 1140+0x41"),
                *scan(
                    pod=second_pod,
                    process=second_slot,
                    drift=f"0+0x19 {current}+0x41",
                    at="0+0x19 1200+60x41",
                ),
            ]
            add(
                f"takeover-{second_pod}-{current}",
                inputs,
                [
                    expected(DRIFT_ALERT, fired, "20m"),
                    expected(SCAN_UNKNOWN, 0, "20m"),
                    expected(DRIFT_ALERT, fired, "40m"),
                ],
            )
    add(
        "inclusive-scan-age",
        [up, *scan(at="1800+0x60")],
        [expected(SCAN_UNKNOWN, 0, "33m"), expected(SCAN_UNKNOWN, 1, "40m")],
    )

    for label, evidence, failed, stale, unknown in (
        ("unconfigured", [], 0, 0, 0),
        ("unknown", [("status_unreadable", "1")], 0, 0, 1),
        ("failed", [("status_unreadable", "0"), ("last_run_ok", "0")], 1, 0, 0),
        (
            "fresh",
            [
                ("status_unreadable", "0"),
                ("last_run_ok", "1"),
                ("last_success_age_seconds", "30"),
            ],
            0,
            0,
            0,
        ),
        (
            "stale",
            [
                ("status_unreadable", "0"),
                ("last_run_ok", "1"),
                ("last_success_age_seconds", "14400"),
            ],
            0,
            1,
            0,
        ),
    ):
        inputs = [series(REFRESH + name, f"{value}+0x60") for name, value in evidence]
        add(
            "refresh-" + label,
            inputs,
            [
                expected(REFRESH_FAILED, failed),
                expected(REFRESH_STALE, stale),
                expected(REFRESH_UNKNOWN, unknown),
            ],
        )
    for name, value, alert in (
        ("last_run_ok", "0", REFRESH_FAILED),
        ("status_unreadable", "1", REFRESH_UNKNOWN),
    ):
        add(
            "refresh-healthy-peer-cannot-hide-" + name,
            [
                series(REFRESH + "last_run_ok", "1+0x60"),
                series(REFRESH + "status_unreadable", "0+0x60"),
                series(REFRESH + name, f"{value}+0x60", pod="worker-b"),
            ],
            [expected(alert, 1)],
        )

    for label, stamp, retained, fired in (
        ("no-event", "0+0x60", "1+0x60", 0),
        ("retirement-only", "0+0x60", "10+0x19 0+0x41", 0),
        ("new-failure", "0+0x19 1200+0x41", "0+0x19 1+0x41", 1),
        ("retirement-masks-population", "0+0x19 1200+0x41", "1+0x60", 1),
        ("retirement-outweighs-failure", "0+0x19 1200+0x41", "100+0x19 1+0x41", 1),
        ("child-restart-after-observation", "0+0x19 1200+0x1 0+0x40", "1+0x60", 1),
    ):
        add(
            "notification-" + label,
            [
                series(TERMINAL, stamp),
                series("gpu_fault_notification_total", retained, status="FAILED"),
            ],
            [
                expected(NOTIFY_ALERT, 0, "24m"),
                expected(NOTIFY_ALERT, fired, "27m"),
                expected(NOTIFY_ALERT, 0, "35m"),
            ],
        )
    add(
        "notification-whole-pod-restart",
        [
            series(TERMINAL, "0+0x19 1200+0x1 stale"),
            series(TERMINAL, " ".join(["_"] * 22) + " 0+0x40", pod="worker-new"),
        ],
        [expected(NOTIFY_ALERT, 1, "27m"), expected(NOTIFY_ALERT, 0, "35m")],
    )
    add(
        "periodic-job-survives-target-job",
        [
            series(
                "gpu_fault_periodic_cleanup_rows_total",
                "0+0x30 1+0x30",
                process="0",
                periodic_job="stale_fence_remote_commands",
            )
        ],
        [expected(SWEEP, 1)],
    )
    add(
        "periodic-error-keeps-task-name",
        [
            series(
                "gpu_fault_periodic_job_error_last_seen_timestamp_seconds",
                "0+0x30 1860+0x30",
                periodic_job="cleanup",
            )
        ],
        [
            {
                "expr": 'count(ALERTS{alertname="GpuFaultPeriodicServiceErrors",alertstate="firing",periodic_job="cleanup"}) or vector(0)',
                "eval_time": "40m",
                "exp_samples": [{"labels": "{}", "value": 1}],
            }
        ],
    )
    return result


def test_changed_operator_rules_are_exact_amp_equivalents() -> None:
    amp = rule_map("deploy/observability/amp-rules.yaml")
    operator = rule_map(
        "deploy/control-plane/regional/processor-alerts.yaml", wrapped=True
    )
    assert {name: amp[name] for name in MIRRORED} == {
        name: operator[name] for name in MIRRORED
    }


def test_new_metric_contract_is_registered_and_reaches_amp() -> None:
    for name in (SCAN, MAX_AGE, DRIFT, MISMATCH):
        assert STRATEGIES[name] is Strategy.PER_PROCESS
    assert STRATEGIES[TERMINAL] is Strategy.MAX
    documents = yaml.safe_load_all(
        (ROOT / "deploy/observability/adot-control-plane.yaml").read_text()
    )
    config = next(doc for doc in documents if doc["kind"] == "ConfigMap")
    scrape = yaml.safe_load(config["data"]["collector.yaml"])["receivers"][
        "prometheus"
    ]["config"]["scrape_configs"][0]
    keep = next(
        item["regex"]
        for item in scrape["metric_relabel_configs"]
        if item["action"] == "keep"
    )
    assert all(re.fullmatch(keep, name) for name in (SCAN, MAX_AGE, TERMINAL)), (
        "the production ADOT filter must retain every new evidence metric"
    )
    assert not scrape.get("honor_labels", False), (
        "target identity remains authoritative"
    )


@pytest.mark.parametrize("inputs,checks", cases())
def test_observability_evidence_with_real_promql(
    tmp_path: Path, inputs, checks
) -> None:
    promtool = shutil.which(os.environ.get("PROMTOOL", "promtool"))
    assert promtool is not None, "set PROMTOOL to the pinned local promtool"
    source = rule_map("deploy/observability/amp-rules.yaml")
    names = MIRRORED | set(COUNTER_ALERTS.values())
    rule_file = tmp_path / "rules.yaml"
    rule_file.write_text(
        yaml.safe_dump(
            {
                "groups": [
                    {
                        "name": "observability",
                        "rules": [source[name] for name in sorted(names)],
                    }
                ]
            },
            sort_keys=False,
        )
    )
    test_file = tmp_path / "cases.yaml"
    test_file.write_text(
        yaml.safe_dump(
            {
                "rule_files": [str(rule_file)],
                "evaluation_interval": "1m",
                "tests": [
                    {
                        "interval": "1m",
                        "input_series": inputs,
                        "promql_expr_test": checks,
                    }
                ],
            },
            sort_keys=False,
        )
    )
    result = subprocess.run(
        [promtool, "test", "rules", str(test_file)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
