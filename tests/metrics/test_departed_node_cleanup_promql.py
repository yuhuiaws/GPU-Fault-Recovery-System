"""``GpuFaultDepartedNodeCleanupStuck`` evaluated by Prometheus itself (promtool).

The departed-node cleanup ends a departed node's records minutes after the
departure is proven; a ledger line still PENDING or REFUSED two hours later is
residue nobody is ending. The rule reads the age of the oldest open line per
GPU cluster and state, which every worker renders alike from the store, so the
cases pin what an operator sees: one warning per cluster and state however
many workers render it, none below two hours or once the line is gone.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.departed_node_cleanup import ledger_metric_lines
from gpu_fault.departed_nodes import (
    DepartedNodeCleanupRecord,
    DepartedNodeCleanupStatus,
)
from tests.departed_nodes._support import ago, hyperpod_departure

ROOT = Path(__file__).resolve().parents[2]
ALERT = "GpuFaultDepartedNodeCleanupStuck"
AGE = "gpu_fault_departed_node_cleanup_oldest_open_age_seconds"
BASE = {
    "control_plane_cluster": "control-a",
    "region": "test-region",
    "job": "gpu-fault-control-plane",
    "namespace": "control",
    "service_role": "gpu-fault-control-worker",
}
FIRING = f'ALERTS{{alertname="{ALERT}",alertstate="firing"}}'


def _rule() -> dict[str, Any]:
    document = yaml.safe_load(
        (ROOT / "deploy/observability/amp-rules.yaml").read_text(encoding="utf-8")
    )
    [rule] = [
        rule
        for group in document["groups"]
        for rule in group["rules"]
        if rule.get("alert") == ALERT
    ]
    return dict(rule)


def _series(
    values: str, *, status: str = "REFUSED", pod: str = "worker-a"
) -> dict[str, str]:
    labels = {
        **BASE,
        "pod": pod,
        "instance": f"{pod}:8000",
        "cluster_id": "gpu-a",
        "status": status,
    }
    rendered = ",".join(f"{key}={json.dumps(value)}" for key, value in labels.items())
    return {"series": f"{AGE}{{{rendered}}}", "values": values}


def _firing(*statuses: str, at: str = "20m") -> dict[str, Any]:
    return {
        "expr": f"count by (cluster_id, status) ({FIRING})",
        "eval_time": at,
        "exp_samples": [
            {"labels": f'{{cluster_id="gpu-a",status="{status}"}}', "value": 1}
            for status in statuses
        ],
    }


def _silent(at: str = "20m") -> dict[str, Any]:
    return {
        "expr": f"count({FIRING}) or vector(0)",
        "eval_time": at,
        "exp_samples": [{"labels": "{}", "value": 0}],
    }


CASES = [
    pytest.param([_series("6900+60x40")], [_firing("REFUSED")], id="refused-past-2h"),
    pytest.param([_series("3600+60x40")], [_silent()], id="under-2h"),
    pytest.param(
        [_series("6900+60x40"), _series("6900+60x40", pod="worker-b")],
        [_firing("REFUSED")],
        id="every-worker-renders-it-one-alert",
    ),
    pytest.param(
        [_series("7300+60x40", status="PENDING"), _series("7300+60x40")],
        [_firing("PENDING", "REFUSED")],
        id="one-alert-per-state",
    ),
    pytest.param(
        [_series("7300+60x8")], [_silent()], id="the-line-ended-the-alert-resolves"
    ),
    pytest.param(
        [_series("7140+60x40")],
        [_silent(at="4m"), _firing("REFUSED", at="10m")],
        id="fires-after-its-for-duration",
    ),
]


@pytest.fixture(scope="module")
def promtool() -> str:
    resolved = shutil.which(os.environ.get("PROMTOOL", "promtool"))
    if resolved is None:
        pytest.fail("promtool is required; set PROMTOOL to the pinned local binary")
    return resolved


@pytest.mark.parametrize(("inputs", "checks"), CASES)
def test_the_stuck_alert_with_real_promql(
    tmp_path: Path, promtool: str, inputs: list[dict[str, str]], checks: list[dict]
) -> None:
    rule = _rule()
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        yaml.safe_dump({"groups": [{"name": "departed", "rules": [rule]}]}),
        encoding="utf-8",
    )
    cases = tmp_path / "cases.yaml"
    cases.write_text(
        yaml.safe_dump(
            {
                "rule_files": [str(rules)],
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
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [promtool, "test", "rules", str(cases)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert rule["labels"] == {"severity": "warning"}, "a warning, never a page"


def _keep_regexes() -> list[str]:
    documents = yaml.safe_load_all(
        (ROOT / "deploy/observability/adot-control-plane.yaml").read_text(
            encoding="utf-8"
        )
    )
    config_map = next(item for item in documents if item["kind"] == "ConfigMap")
    config = yaml.safe_load(config_map["data"]["collector.yaml"])
    jobs = config["receivers"]["prometheus"]["config"]["scrape_configs"]
    job = next(item for item in jobs if item["job_name"] == "gpu-fault-control-plane")
    return [
        item["regex"]
        for item in job["metric_relabel_configs"]
        if item["action"] == "keep" and item["source_labels"] == ["__name__"]
    ]


def test_every_cleanup_family_survives_the_control_plane_keep_filter() -> None:
    from tests._builders import build_store

    store = build_store()
    world_ = hyperpod_departure(store)
    world_.cleanup.run_once()
    open_line = DepartedNodeCleanupRecord(
        cluster_id="gpu-a",
        node_id="node-a",
        status=DepartedNodeCleanupStatus.REFUSED,
        first_seen_at=ago(hours=3),
        updated_at=ago(hours=3),
    )
    lines = [
        *world_.cleanup.metric_lines(),
        *ledger_metric_lines([open_line], now=ago(seconds=0), enabled=True),
    ]
    names = {line.split()[2] for line in lines if line.startswith("# TYPE ")}
    keeps = _keep_regexes()

    assert AGE in names and len(names) == 7, names
    assert keeps, "ADOT must retain an explicit metric-name allowlist"
    dropped = sorted(
        name for name in names if not any(re.fullmatch(keep, name) for keep in keeps)
    )
    assert dropped == [], f"dropped before AMP, so unalertable: {dropped}"
