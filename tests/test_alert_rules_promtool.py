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
from pathlib import Path

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
    assert "$(PROMTOOL_VERSION)" in install and "sha256sum" in install, (
        "ci-supply-chain-tools must download promtool and verify its checksum"
    )
    target = MAKEFILE.split("\npromtool-check:\n", 1)[1].split("\n\n", 1)[0]
    assert "scripts/check-alert-rules.py" in target
    assert "ci-supply-chain-tools" in target, (
        "the strict CI branch must name the installer"
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


def test_repository_rules_pass_promtool_when_it_is_available() -> None:
    """Real check when promtool is on PATH or PROMTOOL points at it; else the
    advisory path must say so and exit 0 (CI installs it, so CI is strict)."""

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
        env={**os.environ, "CI": ""},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report["files"] == 2
    assert report["status"] in {"checked", "skipped"}
    if report["status"] == "checked":
        assert report["rules"] >= 60
