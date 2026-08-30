from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from importlib.resources import files
from typing import Any, Callable, Mapping, Sequence

from gpu_fault.admin_bootstrap_common import BootstrapError


TOOL_MANIFEST = "data/deploy-host-tools.json"


def load_deploy_host_tool_manifest() -> dict[str, Any]:
    document = json.loads(
        files("gpu_fault").joinpath(TOOL_MANIFEST).read_text(encoding="utf-8")
    )
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise BootstrapError("deploy-host tool manifest schema_version must be 1")
    python = document.get("python")
    tools = document.get("tools")
    if not isinstance(python, dict) or not isinstance(tools, list) or not tools:
        raise BootstrapError("deploy-host tool manifest is incomplete")
    return document


def _version_output(stdout: str, stderr: str) -> str:
    lines = [line.strip() for line in (stdout + "\n" + stderr).splitlines()]
    return next((line[:500] for line in lines if line), "")


def _json_version(stdout: str, field: str) -> str:
    try:
        value: object = json.loads(stdout)
    except json.JSONDecodeError:
        return ""
    for part in field.split("."):
        if not isinstance(value, dict):
            return ""
        value = value.get(part)
    return str(value)[:500] if value is not None else ""


def deploy_host_dependency_report(
    *,
    manifest: Mapping[str, Any] | None = None,
    which: Callable[[str], str | None] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    source = dict(manifest or load_deploy_host_tool_manifest())
    active_which = which or shutil.which
    active_runner = runner or subprocess.run
    python = dict(source["python"])
    expected_python = (int(python["major"]), int(python["minor"]))
    actual_python = (sys.version_info.major, sys.version_info.minor)
    tool_reports = []
    for raw in source["tools"]:
        if not isinstance(raw, dict):
            raise BootstrapError("deploy-host tool manifest contains a non-object")
        name = str(raw.get("name") or "")
        executable = str(raw.get("executable") or "")
        command = raw.get("command")
        if (
            not name
            or not executable
            or not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise BootstrapError("deploy-host tool manifest contains an invalid tool")
        path = active_which(executable)
        if path is None:
            tool_reports.append(
                {
                    "name": name,
                    "executable": executable,
                    "status": "MISSING",
                    "version": "",
                }
            )
            continue
        completed = active_runner(
            list(command),
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        version_field = raw.get("version_json_field")
        version = (
            _json_version(completed.stdout or "", str(version_field))
            if version_field
            else ""
        )
        tool_reports.append(
            {
                "name": name,
                "executable": path,
                "status": "PASS" if completed.returncode == 0 else "FAILED",
                "version": version
                or _version_output(
                    completed.stdout or "",
                    completed.stderr or "",
                ),
            }
        )
    return {
        "schema_version": 1,
        "healthy": actual_python == expected_python
        and all(item["status"] == "PASS" for item in tool_reports),
        "python": {
            "expected": f"{expected_python[0]}.{expected_python[1]}",
            "actual": f"{actual_python[0]}.{actual_python[1]}",
            "status": "PASS" if actual_python == expected_python else "FAILED",
        },
        "tools": tool_reports,
    }


def validate_bootstrap_dependencies() -> dict[str, Any]:
    report = deploy_host_dependency_report()
    failures = [item["name"] for item in report["tools"] if item["status"] != "PASS"]
    if report["python"]["status"] != "PASS":
        failures.insert(0, f"python-{report['python']['expected']}")
    if failures:
        raise BootstrapError(
            "deployment host dependency check failed: " + ", ".join(failures)
        )
    return report


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate deployment-host tools")
    parser.add_argument("--output", choices=("json", "text"), default="json")
    options = parser.parse_args(arguments)
    try:
        report = validate_bootstrap_dependencies()
    except BootstrapError as exc:
        print(f"deploy-host-check: {exc}", file=sys.stderr)
        return 2
    if options.output == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("deployment host dependencies: PASS")
        for item in report["tools"]:
            print(f"{item['name']}: {item['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
