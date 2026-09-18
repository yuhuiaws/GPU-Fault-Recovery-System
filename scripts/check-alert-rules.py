"""Run ``promtool check rules`` over the repository's alert rule files.

``scripts/verify-regional-alerting.py`` proves that every alert has a runbook,
a description and an aggregation that will not fan out per replica; it does
not parse PromQL. This script hands the same two files to Prometheus's own
``promtool``, after unwrapping the Kubernetes ``PrometheusRule`` envelope that
``promtool`` does not understand, so a malformed expression fails the build
rather than the first Prometheus reload.

Exit codes: 0 when every requested check passes, 1 when promtool rejects a file,
2 when the required tool is unavailable or a file carries no rule groups.
``--tool-only`` requires a version pin and checks only the executable, before
local test/release gates start expensive work. It uses only the Python standard
library and never installs tools.
``--pytest-files`` additionally requires complete, passing behavioral receipts;
this mode never treats missing tools or skipped tests as successful evidence.
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

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.pytest_result_identity import (
    CI_CONTEXT_ENV,
    parse_pytest_receipt,
    source_identity,
)

ROOT = Path(__file__).resolve().parents[1]
RULE_FILES = (
    ROOT / "deploy/observability/amp-rules.yaml",
    ROOT / "deploy/control-plane/regional/processor-alerts.yaml",
)
RULES_FOUND = re.compile(r"SUCCESS: (\d+) rules found")


class AlertRulesError(RuntimeError):
    pass


def require_promtool(candidate: str, *, version: str = "") -> str:
    """Resolve one executable and, when pinned, require its exact version."""

    resolved = shutil.which(candidate)
    if resolved is None:
        raise AlertRulesError(
            "promtool is required and must be executable: "
            "run make ci-supply-chain-tools or set PROMTOOL to the pinned binary"
        )
    resolved = str(Path(resolved).absolute())
    if version:
        try:
            banner = subprocess.run(
                [resolved, "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
            raise AlertRulesError(
                "promtool version probe failed; run make ci-supply-chain-tools "
                "or set PROMTOOL to the pinned binary"
            ) from exc
        reported = re.search(
            r"^promtool, version ([^\s,]+)",
            banner.stdout + banner.stderr,
            re.MULTILINE,
        )
        if banner.returncode or reported is None or reported.group(1) != version:
            raise AlertRulesError(
                f"promtool must report version {version}; "
                "run make ci-supply-chain-tools or set PROMTOOL to the pinned binary"
            )
    return resolved


def rule_documents(paths: Sequence[Path]) -> list[tuple[str, dict[str, Any]]]:
    """``(name, {"groups": [...]})`` per file, unwrapping ``PrometheusRule``."""

    import yaml  # type: ignore[import-untyped,unused-ignore]

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

    import yaml  # type: ignore[import-untyped,unused-ignore]

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


def check_behavior_tests(
    paths: Sequence[str], *, promtool: str, root: Path = ROOT
) -> int:
    expected = sorted(paths)
    if (
        not expected
        or len(set(expected)) != len(expected)
        or any(
            not name.startswith("tests/")
            or not name.endswith(".py")
            or ".." in Path(name).parts
            or not (root / name).is_file()
            for name in expected
        )
    ):
        raise AlertRulesError("PromQL pytest targets must be complete test files")
    with tempfile.TemporaryDirectory(prefix="gpu-fault-promql-") as directory:
        report = Path(directory) / "pytest-results.json"
        environment = {
            **os.environ,
            "PROMTOOL": promtool,
            "GPU_FAULT_TEST_POSTGRES_URL": "",
            "PYTEST_GPU_FAULT_CASE_REPORT": str(report),
        }
        for name in (
            "PYTEST_ADDOPTS",
            "PYTEST_GPU_FAULT_PARTITION_COUNT",
            "PYTEST_GPU_FAULT_PARTITION_INDEX",
            CI_CONTEXT_ENV,
        ):
            environment.pop(name, None)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *paths,
                "-n",
                "0",
                "-o",
                "addopts=",
                "-p",
                "tools.pytest_case_reporter",
                f"--basetemp={directory}/pytest",
            ],
            cwd=root,
            env=environment,
            check=False,
        )
        if completed.returncode:
            return 1
        try:
            value = json.loads(report.read_text(encoding="utf-8"))
            receipt = parse_pytest_receipt(
                value,
                root=root,
                expected_identity=source_identity(root),
                require_session=True,
                require_passed=True,
            )
            session = value["session"]
            selection = session.get("selection")
            if (
                session.get("collection_errors") != []
                or session.get("collection_skips") != []
                or session.get("collected_files") != expected
                or receipt.discovered_nodeids != frozenset(receipt.records)
                or not isinstance(selection, dict)
                or selection.get("targets") != expected
                or selection.get("partition") is not None
                or selection.get("keyword") != ""
                or selection.get("markexpr") != ""
                or selection.get("deselect") != []
            ):
                raise ValueError("PromQL pytest did not complete the full selection")
        except (OSError, ValueError) as exc:
            raise AlertRulesError(f"PromQL pytest evidence is invalid: {exc}") from exc
    print(
        json.dumps(
            {"status": "tested", "files": len(paths), "tests": len(receipt.records)}
        )
    )
    return 0


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
    parser.add_argument(
        "--pytest-files",
        nargs="+",
        default=(),
        help="behavioral test files that must complete without skips",
    )
    parser.add_argument(
        "--tool-only",
        action="store_true",
        help="require the pinned executable without checking rules or running tests",
    )
    options = parser.parse_args(arguments)
    if (options.pytest_files or options.tool_only) and not options.version:
        print(
            "tool and behavioral preflights require the pinned --version",
            file=sys.stderr,
        )
        return 2
    if options.tool_only and options.pytest_files:
        print("--tool-only cannot be combined with --pytest-files", file=sys.stderr)
        return 2
    try:
        resolved = require_promtool(options.promtool, version=options.version)
        if options.tool_only:
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "promtool": resolved,
                        "version": options.version,
                    }
                )
            )
            return 0
        result = check(RULE_FILES, promtool=resolved)
        if result or not options.pytest_files:
            return result
        return check_behavior_tests(options.pytest_files, promtool=resolved)
    except AlertRulesError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
