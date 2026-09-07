"""Run ``promtool check rules`` over the repository's alert rule files.

``scripts/verify-regional-alerting.py`` proves that every alert has a runbook,
a description and an aggregation that will not fan out per replica; it does
not parse PromQL. This script hands the same two files to Prometheus's own
``promtool``, after unwrapping the Kubernetes ``PrometheusRule`` envelope that
``promtool`` does not understand, so a malformed expression fails the build
rather than the first Prometheus reload.

Exit codes: 0 when every file passes (or promtool is absent outside CI, which
prints a skip), 1 when promtool rejects a file, 2 when promtool is required
(``CI`` set) but missing or when a file carries no rule groups.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
RULE_FILES = (
    ROOT / "deploy/observability/amp-rules.yaml",
    ROOT / "deploy/control-plane/regional/processor-alerts.yaml",
)
RULES_FOUND = re.compile(r"SUCCESS: (\d+) rules found")


class AlertRulesError(RuntimeError):
    pass


def rule_documents(paths: Sequence[Path]) -> list[tuple[str, dict[str, Any]]]:
    """``(name, {"groups": [...]})`` per file, unwrapping ``PrometheusRule``."""

    documents: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        groups: object = None
        if isinstance(loaded, dict):
            if isinstance(loaded.get("groups"), list):
                groups = loaded["groups"]
            elif isinstance(loaded.get("spec"), dict) and isinstance(
                loaded["spec"].get("groups"), list
            ):
                groups = loaded["spec"]["groups"]
        if not isinstance(groups, list) or not groups:
            raise AlertRulesError(f"{path.name} carries no Prometheus rule groups")
        documents.append((path.name, {"groups": groups}))
    return documents


def check(paths: Sequence[Path], *, promtool: Path | str) -> int:
    """Write each unwrapped document to a temp file and run promtool once each."""

    total_rules = 0
    failures = 0
    with tempfile.TemporaryDirectory(prefix="gpu-fault-promtool-") as workdir:
        for name, document in rule_documents(paths):
            target = Path(workdir) / name
            target.write_text(
                yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [str(promtool), "check", "rules", str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            output = completed.stdout + completed.stderr
            found = RULES_FOUND.search(output)
            total_rules += int(found.group(1)) if found else 0
            if completed.returncode:
                failures += 1
                print(f"promtool rejected {name}:\n{output}", file=sys.stderr)
            else:
                print(f"promtool: {name} ok")
    print(
        json.dumps(
            {
                "status": "checked",
                "files": len(paths),
                "rules": total_rules,
                "failures": failures,
            }
        )
    )
    return 1 if failures else 0


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--promtool",
        default=os.environ.get("PROMTOOL", "promtool"),
        help="promtool executable (default: $PROMTOOL or promtool on PATH)",
    )
    parser.add_argument(
        "--version",
        default="",
        help="when set, the promtool binary must report this version",
    )
    options = parser.parse_args(arguments)
    resolved = (
        shutil.which(options.promtool)
        if os.sep not in options.promtool
        else (options.promtool if os.access(options.promtool, os.X_OK) else None)
    )
    if resolved is None:
        if os.environ.get("CI"):
            print(
                "promtool is required in CI: run make ci-supply-chain-tools",
                file=sys.stderr,
            )
            return 2
        print(
            "promtool is not installed; PromQL check skipped, CI runs promtool "
            f"{options.version or '(pinned in the Makefile)'}"
        )
        print(json.dumps({"status": "skipped", "files": len(RULE_FILES), "rules": 0}))
        return 0
    if options.version:
        banner = subprocess.run(
            [resolved, "--version"], capture_output=True, text=True, check=False
        )
        reported = banner.stdout + banner.stderr
        if f"version {options.version}" not in reported:
            print(
                f"promtool at {resolved} is not version {options.version}: "
                f"{reported.strip().splitlines()[0] if reported.strip() else '?'}",
                file=sys.stderr,
            )
            return 2
    try:
        return check(RULE_FILES, promtool=resolved)
    except AlertRulesError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
