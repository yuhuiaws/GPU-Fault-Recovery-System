"""The alert rule files are checked by promtool, not only by our static checker.

`scripts/verify-regional-alerting.py` proves runbook links, annotations and
aggregation shape; it cannot parse PromQL. `scripts/check-alert-rules.py` hands
both rule files to `promtool check rules`, unwrapping the Kubernetes
PrometheusRule envelope first, so a syntax error in an alert expression fails
the build instead of the first Prometheus reload.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
MODULE = lazy_script_module(ROOT / "scripts/check-alert-rules.py")
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")


def _rule(name: str, expr: str) -> dict:
    return {"alert": name, "expr": expr, "labels": {"severity": "warning"}}


def test_prometheusrule_envelopes_are_unwrapped_and_plain_files_pass_through(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain.yaml"
    plain.write_text(
        yaml.safe_dump({"groups": [{"name": "g", "rules": [_rule("A", "up == 0")]}]}),
        encoding="utf-8",
    )
    wrapped = tmp_path / "wrapped.yaml"
    wrapped.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "monitoring.coreos.com/v1",
                "kind": "PrometheusRule",
                "metadata": {"name": "x", "namespace": "ns"},
                "spec": {"groups": [{"name": "h", "rules": [_rule("B", "up == 1")]}]},
            }
        ),
        encoding="utf-8",
    )

    documents = MODULE.rule_documents([plain, wrapped])

    assert [name for name, _ in documents] == ["plain.yaml", "wrapped.yaml"]
    assert documents[0][1] == {
        "groups": [{"name": "g", "rules": [_rule("A", "up == 0")]}]
    }
    assert documents[1][1] == {
        "groups": [{"name": "h", "rules": [_rule("B", "up == 1")]}]
    }


def test_a_document_without_rule_groups_is_rejected(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.yaml"
    bogus.write_text("kind: ConfigMap\n", encoding="utf-8")

    try:
        MODULE.rule_documents([bogus])
    except MODULE.AlertRulesError as exc:
        assert "bogus.yaml" in str(exc)
    else:
        raise AssertionError("a file with no rule groups must be refused")


def test_check_runs_promtool_once_per_document_with_unwrapped_files(
    tmp_path: Path,
) -> None:
    fake = tmp_path / "promtool"
    log = tmp_path / "calls.json"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'printf \'%s\\n\' "$@" >> "{log}"\n'
        'for f in "${@:3}"; do grep -q "^groups:" "$f" || exit 7; done\n'
        "echo SUCCESS\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    wrapped = tmp_path / "wrapped.yaml"
    wrapped.write_text(
        yaml.safe_dump(
            {
                "kind": "PrometheusRule",
                "spec": {"groups": [{"name": "h", "rules": [_rule("B", "up == 1")]}]},
            }
        ),
        encoding="utf-8",
    )

    exit_code = MODULE.check([wrapped], promtool=fake)

    assert exit_code == 0
    calls = log.read_text(encoding="utf-8").split()
    assert calls[:2] == ["check", "rules"]
    assert len(calls) == 3 and calls[2].endswith("wrapped.yaml")


def test_repository_rule_files_are_the_two_alert_sources() -> None:
    assert [path.relative_to(ROOT).as_posix() for path in MODULE.RULE_FILES] == [
        "deploy/observability/amp-rules.yaml",
        "deploy/control-plane/regional/processor-alerts.yaml",
    ]


def test_promtool_is_pinned_by_version_and_checksum_and_gated_like_the_other_tools() -> (
    None
):
    version = re.search(
        r"^PROMTOOL_VERSION \?= (\d+\.\d+\.\d+)$", MAKEFILE, re.MULTILINE
    )
    digest = re.search(
        r"^PROMTOOL_SHA256_LINUX_AMD64 \?= ([0-9a-f]{64})$", MAKEFILE, re.MULTILINE
    )
    assert version is not None, "PROMTOOL_VERSION is not pinned in the Makefile"
    assert digest is not None, "the promtool tarball checksum is not pinned"
    install = MAKEFILE.split("\nci-supply-chain-tools:\n", 1)[1].split("\n\n", 1)[0]
    assert "scripts/setup_supply_chain_tools.py" in install, (
        "ci-supply-chain-tools must use the shared pinned installer"
    )
    assert '--promtool-version "$(PROMTOOL_VERSION)"' in install, (
        "the installer must receive Make's authoritative version pin"
    )
    assert '--promtool-sha256 "$(PROMTOOL_SHA256_LINUX_AMD64)"' in install, (
        "the installer must receive Make's authoritative archive checksum"
    )
    target = MAKEFILE.split("\npromtool-check:\n", 1)[1].split("\n\n", 1)[0]
    assert "scripts/check-alert-rules.py" in target
    assert '--version "$(PROMTOOL_VERSION)"' in target, (
        "the behavioral gate must use the same pin as the tool preflight"
    )
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    runs = [
        step["run"]
        for step in ci["jobs"]["static"]["steps"]
        if isinstance(step.get("run"), str)
    ]
    assert "make promtool-check PYTHON=python" in runs
    assert runs.index("make promtool-check PYTHON=python") > runs.index(
        "make ci-supply-chain-tools PYTHON=python"
    )
    step = next(
        item
        for item in ci["jobs"]["static"]["steps"]
        if item.get("run") == "make promtool-check PYTHON=python"
    )
    assert not step.get("if"), (
        "native PromQL checks cannot be conditional or cache-reused"
    )


def test_repository_rules_pass_promtool_when_it_is_available() -> None:
    """The static owner and full local run must use a real, available tool."""

    candidate = os.environ.get("PROMTOOL") or "promtool"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check-alert-rules.py"),
            "--promtool",
            candidate,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "CI": "true"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report["files"] == 2
    assert report["status"] == "checked", (
        "missing native checks are not successful evidence"
    )
    assert report["rules"] >= 60


# The processor queue depth rules are ratios of the caps admission enforces,
# which /metrics exports next to the depths. They compared with 7000 / 9000 /
# 700 -- 70 / 90 / 70 percent of the code defaults -- while the regional
# manifests render 65536 / 4096, so they fired at 11-17 percent of the real
# caps and "exceeds 70 percent" was false. Every case below puts the depth on
# one side of a boundary, under the rendered caps or under caps that move it.
QUEUE_ALERTS = (
    "GpuFaultProcessorQueueDepthHigh",
    "GpuFaultProcessorQueueDepthCritical",
    "GpuFaultProcessorClusterBacklogHigh",
    "GpuFaultProcessorClusterBacklogCritical",
)
QUEUE_SITE = {"control_plane_cluster": "control-a", "region": "test-region"}
HIGH, CRITICAL = QUEUE_ALERTS[0], QUEUE_ALERTS[1]
CLUSTER_HIGH, CLUSTER_CRITICAL = QUEUE_ALERTS[2], QUEUE_ALERTS[3]


def _queue_rules(relative: str) -> dict[str, dict]:
    [(_, document)] = MODULE.rule_documents([ROOT / relative])
    return {
        rule["alert"]: rule
        for group in document["groups"]
        for rule in group["rules"]
        if rule.get("alert") in QUEUE_ALERTS
    }


def _queue_series(
    pod: str,
    *,
    depth: int | str,
    clusters: dict[str, int | str],
    limit: int | str = 65536,
    reserve: int | str = 8192,
    cluster_limit: int | str = 4096,
    cluster_reserve: int | str = 512,
) -> list[dict[str, str]]:
    """One replica's queue view and caps, labelled the way ADOT labels them.

    The defaults are the caps the regional manifests render.
    """

    target = {
        **QUEUE_SITE,
        "job": "gpu-fault-control-plane",
        "namespace": "gpu-fault-system",
        "pod": pod,
        "instance": f"{pod}:8000",
        "service_role": "gpu-fault-control-worker",
    }

    def one(metric: str, values: int | str, **labels: str) -> dict[str, str]:
        tags = {**target, **labels}
        rendered = ",".join(f'{key}="{value}"' for key, value in tags.items())
        return {
            "series": f"{metric}{{{rendered}}}",
            "values": values if isinstance(values, str) else f"{values}+0x30",
        }

    return [
        one("gpu_fault_processor_queue_depth", depth),
        one("gpu_fault_processor_queue_depth_limit", limit),
        one("gpu_fault_processor_queue_fault_reserved_depth", reserve),
        one("gpu_fault_processor_cluster_queue_depth_limit", cluster_limit),
        one("gpu_fault_processor_cluster_queue_fault_reserved_depth", cluster_reserve),
        *(
            one("gpu_fault_processor_cluster_queue_depth", value, cluster_id=cluster)
            for cluster, value in clusters.items()
        ),
    ]


def _two_replicas(**values) -> list[dict[str, str]]:
    """A worker and an ingress replica publishing the same Store-derived depths."""

    return [
        *_queue_series("gpu-fault-control-worker-0", **values),
        *_queue_series("gpu-fault-api-ha-0", **values),
    ]


def _queue_alert(rule: dict, value: float, cluster: str | None) -> dict:
    labels = {**QUEUE_SITE, "severity": rule["labels"]["severity"]}
    rendered = {
        "{{ $value | humanizePercentage }}": f"{value * 100:.4g}%",
        '{{ $value | printf "%.0f" }}': f"{value:.0f}",
    }
    if cluster is not None:
        labels["cluster_id"] = cluster
        rendered["{{ $labels.cluster_id }}"] = cluster
    annotations = {}
    for key, text in rule["annotations"].items():
        for template, value_text in rendered.items():
            text = str(text).replace(template, value_text)
        annotations[key] = text
    return {"exp_labels": labels, "exp_annotations": annotations}


QUEUE_CASES = [
    pytest.param(
        _two_replicas(depth=9001, clusters={"a": 701}),
        [("10m", {})],
        id="the-old-9000-and-700-thresholds-are-13-and-17-percent-of-the-caps",
    ),
    pytest.param(
        _two_replicas(depth=45875, clusters={"a": 2867}),
        [("10m", {})],
        id="just-below-70-percent-nothing-fires",
    ),
    pytest.param(
        _two_replicas(depth=45876, clusters={}),
        [("10m", {HIGH: [(45876 / 65536, None)]})],
        id="regional-just-above-70-percent-warns",
    ),
    pytest.param(
        _two_replicas(depth=58982, clusters={}),
        [("10m", {HIGH: [(58982 / 65536, None)]})],
        id="regional-just-below-90-percent-only-warns",
    ),
    pytest.param(
        _two_replicas(depth=58983, clusters={}),
        [("10m", {HIGH: [(58983 / 65536, None)], CRITICAL: [(58983 / 65536, None)]})],
        id="regional-just-above-90-percent-pages",
    ),
    pytest.param(
        _two_replicas(depth=9001, limit=10000, reserve=4000, clusters={}),
        [("10m", {HIGH: [(0.9001, None)], CRITICAL: [(0.9001, None)]})],
        id="the-code-default-cap-moves-the-regional-boundaries-down",
    ),
    pytest.param(
        _two_replicas(depth=50000, limit=131072, reserve=16384, clusters={}),
        [("10m", {})],
        id="a-larger-regional-cap-moves-the-boundary-up",
    ),
    pytest.param(
        [
            *_queue_series("gpu-fault-control-worker-0", depth=50000, clusters={}),
            *_queue_series(
                "gpu-fault-control-worker-1",
                depth=50000,
                limit=131072,
                reserve=16384,
                clusters={},
            ),
        ],
        [("10m", {HIGH: [(50000 / 65536, None)]})],
        id="mid-roll-the-tightest-cap-still-enforced-is-the-one-read",
    ),
    pytest.param(
        _two_replicas(
            depth=50000,
            limit="65536+0x10 131072+0x20",
            reserve="8192+0x10 16384+0x20",
            clusters={},
        ),
        [("10m", {HIGH: [(50000 / 65536, None)]}), ("20m", {})],
        id="raising-the-regional-cap-resolves-the-warning",
    ),
    pytest.param(
        _two_replicas(depth=3000, clusters={"a": 2868, "b": 100}),
        [("10m", {CLUSTER_HIGH: [(2868 / 4096, "a")]})],
        id="cluster-just-above-70-percent-warns-for-that-cluster-only",
    ),
    pytest.param(
        _two_replicas(depth=3600, clusters={"a": 3583}),
        [("10m", {CLUSTER_HIGH: [(3583 / 4096, "a")]})],
        id="one-request-below-the-cluster-fault-reserve-only-warns",
    ),
    pytest.param(
        _two_replicas(depth=3600, clusters={"a": 3584}),
        [
            (
                "10m",
                {CLUSTER_HIGH: [(3584 / 4096, "a")], CLUSTER_CRITICAL: [(3584, "a")]},
            )
        ],
        id="at-the-cluster-fault-reserve-boundary-pages",
    ),
    pytest.param(
        _two_replicas(depth=3100, cluster_reserve=1024, clusters={"a": 3072}),
        [
            (
                "10m",
                {CLUSTER_HIGH: [(3072 / 4096, "a")], CLUSTER_CRITICAL: [(3072, "a")]},
            )
        ],
        id="a-reserve-sized-for-a-larger-cluster-moves-the-page-down",
    ),
    pytest.param(
        _two_replicas(
            depth=3600, cluster_limit=8192, cluster_reserve=1024, clusters={"a": 3584}
        ),
        [("10m", {})],
        id="a-larger-cluster-cap-moves-both-cluster-boundaries-up",
    ),
    pytest.param(
        _two_replicas(depth=3600, clusters={"a": "0+0x9 3584+0x21"}),
        [("11m", {}), ("12m", {CLUSTER_CRITICAL: [(3584, "a")]})],
        id="the-cluster-page-waits-two-minutes-at-the-boundary",
    ),
]


def test_queue_depth_rules_are_the_same_in_both_rule_files() -> None:
    amp = _queue_rules("deploy/observability/amp-rules.yaml")
    operator = _queue_rules("deploy/control-plane/regional/processor-alerts.yaml")

    assert sorted(amp) == sorted(QUEUE_ALERTS), sorted(amp)
    assert operator == amp, "the PrometheusRule copies drifted from the AMP rules"


def test_queue_depth_rules_compare_shares_of_the_exported_caps() -> None:
    for name, rule in _queue_rules("deploy/observability/amp-rules.yaml").items():
        expression = " ".join(str(rule["expr"]).split())
        numbers = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", expression)
        assert [float(number) for number in numbers if float(number) > 1] == [], (
            f"{name} compares with a request count instead of a share: {expression}"
        )
        assert "_queue_depth_limit" in expression, (
            f"{name} does not read the cap admission enforces: {expression}"
        )


@pytest.mark.parametrize(("inputs", "checks"), QUEUE_CASES)
def test_queue_depth_alerts_fire_at_the_configured_boundaries(
    tmp_path: Path, inputs: list[dict[str, str]], checks: list
) -> None:
    promtool = shutil.which(os.environ.get("PROMTOOL", "promtool"))
    assert promtool is not None, "set PROMTOOL to the pinned local promtool"
    rules = _queue_rules("deploy/observability/amp-rules.yaml")
    rule_file = tmp_path / "rules.yaml"
    rule_file.write_text(
        yaml.safe_dump(
            {"groups": [{"name": "queue", "rules": [rules[n] for n in QUEUE_ALERTS]}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    alert_tests = [
        {
            "alertname": alert,
            "eval_time": when,
            "exp_alerts": [
                _queue_alert(rules[alert], value, cluster)
                for value, cluster in firing.get(alert, [])
            ],
        }
        for when, firing in checks
        for alert in QUEUE_ALERTS
    ]
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
                        "alert_rule_test": alert_tests,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [promtool, "test", "rules", str(test_file)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "defect",
    [
        "skip",
        "phase",
        "collection-skip",
        "discovery",
        "filter",
        "target",
        "identity",
        "absent",
    ],
)
def test_behavior_gate_rejects_incomplete_zero_exit_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    root = tmp_path / "repo"
    target = root / "tests/test_rules.py"
    target.parent.mkdir(parents=True)
    target.write_text("def test_rule(): pass\n", encoding="utf-8")
    nodeid = "tests/test_rules.py::test_rule"
    monkeypatch.setattr(MODULE, "source_identity", lambda _root: "a" * 64)
    for name in (
        "PYTEST_ADDOPTS",
        "PYTEST_GPU_FAULT_PARTITION_COUNT",
        "PYTEST_GPU_FAULT_PARTITION_INDEX",
        "PYTEST_GPU_FAULT_CI_CONTEXT",
    ):
        monkeypatch.setenv(name, "untrusted inherited selection")

    def run(command, *, env, **_kwargs):
        assert env["PROMTOOL"] == "/controlled/promtool", (
            "the selected tool is explicit"
        )
        assert env["GPU_FAULT_TEST_POSTGRES_URL"] == "", (
            "PromQL does not need a database"
        )
        assert (
            not {
                "PYTEST_ADDOPTS",
                "PYTEST_GPU_FAULT_PARTITION_COUNT",
                "PYTEST_GPU_FAULT_PARTITION_INDEX",
                "PYTEST_GPU_FAULT_CI_CONTEXT",
            }
            & env.keys()
        ), "a static receipt cannot inherit a coverage shard's selection"
        value = {
            "schema_version": 1,
            "source_identity": "a" * 64,
            "session": {
                "source_identity": "a" * 64,
                "exitstatus": 0,
                "collected_nodeids": [nodeid],
                "discovered_nodeids": [nodeid],
                "collected_files": ["tests/test_rules.py"],
                "collection_errors": [],
                "collection_skips": [],
                "selection": {
                    "targets": ["tests/test_rules.py"],
                    "partition": None,
                    "keyword": "",
                    "markexpr": "",
                    "deselect": [],
                },
            },
            "records": {
                nodeid: {
                    "status": "PASS",
                    "phases": {
                        "setup": "passed",
                        "call": "passed",
                        "teardown": "passed",
                    },
                }
            },
        }
        session = value["session"]
        if defect == "skip":
            value["records"][nodeid]["phases"]["call"] = "skipped"
        elif defect == "phase":
            value["records"][nodeid]["phases"].pop("teardown")
        elif defect == "collection-skip":
            session["collection_skips"] = ["tests/test_skipped.py"]
        elif defect == "discovery":
            session["discovered_nodeids"].append("tests/test_rules.py::test_missing")
        elif defect == "filter":
            session["selection"]["keyword"] = "subset"
        elif defect == "target":
            session["collected_files"] = []
        elif defect == "identity":
            session["source_identity"] = "b" * 64
        if defect != "absent":
            Path(env["PYTEST_GPU_FAULT_CASE_REPORT"]).write_text(
                json.dumps(value), encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE, "subprocess", SimpleNamespace(run=run))
    with pytest.raises(MODULE.AlertRulesError, match="pytest evidence"):
        MODULE.check_behavior_tests(
            ["tests/test_rules.py"], promtool="/controlled/promtool", root=root
        )


@pytest.mark.parametrize("reported,returncode", [("3.14.01", 0), ("3.14.0", 1)])
def test_version_pin_rejects_a_prefix_match_or_failed_probe(
    monkeypatch: pytest.MonkeyPatch, reported: str, returncode: int
) -> None:
    def probe(command, **_kwargs):
        assert command == [sys.executable, "--version"], (
            "an invalid version must stop before syntax or behavioral execution"
        )
        return subprocess.CompletedProcess(
            command, returncode, stdout=f"promtool, version {reported}\n", stderr=""
        )

    monkeypatch.setattr(
        MODULE,
        "subprocess",
        SimpleNamespace(run=probe, SubprocessError=subprocess.SubprocessError),
    )
    assert (
        MODULE.main(
            [
                "--promtool",
                sys.executable,
                "--version",
                "3.14.0",
                "--pytest-files",
                "tests/metrics/test_closed_loop_promql.py",
            ]
        )
        == 2
    ), "only a successful exact-version probe can authorize behavioral tests"


def test_behavior_execution_requires_an_explicit_version_pin() -> None:
    assert (
        MODULE.main(
            [
                "--promtool",
                sys.executable,
                "--pytest-files",
                "tests/metrics/test_closed_loop_promql.py",
            ]
        )
        == 2
    ), "a behavioral gate cannot silently use an arbitrary native tool version"


def test_closed_loop_fixture_fails_instead_of_skipping_without_promtool(
    tmp_path: Path,
) -> None:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("PYTEST_GPU_FAULT_") and name != "PYTEST_ADDOPTS"
    }
    environment["PROMTOOL"] = str(tmp_path / "absent-promtool")
    environment["GPU_FAULT_TEST_POSTGRES_URL"] = ""
    with tempfile.TemporaryDirectory(prefix="qprom-") as directory:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                f"--basetemp={directory}/pytest",
                "tests/metrics/test_closed_loop_promql.py::test_closed_loop_promql[fast]",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 1, "a missing tool must fail a direct full-suite run"
    assert "promtool is required" in output, (
        "the failure identifies the missing prerequisite"
    )
    assert "1 skipped" not in output, "a skipped behavioral case is not acceptance"
