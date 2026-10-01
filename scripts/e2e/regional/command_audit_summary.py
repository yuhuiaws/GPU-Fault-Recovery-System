"""Operator-side parsing of the live CMD protocol audit's printed run summary."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROGRESS_LINE = re.compile(
    r"^(ARMED|PASS GF-REGIONAL-CMD-\d{3}|FAIL GF-REGIONAL-CMD-\d{3}: .*)$"
)


def printed_summary(text: str) -> dict[str, Any]:
    """Parse the run summary from a captured stdout of the in-Pod audit.

    The live audit prints one progress line per case (``PASS <id>`` /
    ``FAIL <id>: <reason>``) before the JSON summary, so an operator who captured
    the Pod's stdout into a file hands ``--write-evidence`` both. Only those
    progress lines may precede the JSON object; anything else fails closed.
    """
    lines = text.splitlines()
    index = 0
    while index < len(lines) and PROGRESS_LINE.match(lines[index]):
        index += 1
    body = "\n".join(lines[index:]).strip()
    if not body.startswith("{"):
        raise SystemExit("--write-evidence expects the printed run summary JSON")
    summary = json.loads(body)
    if not isinstance(summary, dict):
        raise SystemExit("--write-evidence expects a JSON object summary")
    return summary


def write_evidence_main(
    arguments: Any,
    write_evidence_from_summary: Callable[..., list[Path]],
) -> int:
    """``--write-evidence`` operator-side entry: parse, write, report, exit code."""
    if arguments.credentials_stdin or arguments.synthetic_run_id:
        raise SystemExit("credential input is only valid for a live audit")
    if arguments.run_dir is None:
        raise SystemExit("--write-evidence needs --run-dir")
    summary = printed_summary(arguments.write_evidence.read_text(encoding="utf-8"))
    written = write_evidence_from_summary(
        summary, arguments.run_dir, release_id=arguments.release_id
    )
    print(json.dumps({"written": [str(path) for path in written]}, indent=2))
    return 0 if summary.get("verdict") == "PASS" else 1
