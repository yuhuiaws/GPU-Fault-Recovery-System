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
