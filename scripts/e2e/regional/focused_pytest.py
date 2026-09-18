"""Complete, source-bound pytest evidence for local live-runner prerequisites."""

from __future__ import annotations

import ipaddress
import re
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from tools import pytest_result_identity as evidence
from tools.run_fault_test_cases import build_isolated_environment

PYTHON_EXECUTABLE = re.compile(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?$")
PASSED_PHASES = {"setup": "passed", "call": "passed", "teardown": "passed"}


def validate_isolated_postgres_url(value: str) -> None:
    valid = False
    if isinstance(value, str) and not any(
        character.isspace() or ord(character) < 32 for character in value
    ):
        try:
            parsed = urlsplit(value)
            host = parsed.hostname or ""
            loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
            query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
            keys = [name for name, _item in query]
            valid = (
                parsed.scheme in {"postgres", "postgresql"}
                and loopback
                and parsed.port is not None
                and 1 <= parsed.port <= 65535
                and re.fullmatch(r"/gpu_fault_cap005_[0-9a-f]{12}", parsed.path)
                is not None
                and not parsed.fragment
                and len(keys) == len(set(keys))
                and set(keys) <= {"sslmode", "sslrootcert", "connect_timeout"}
            )
        except ValueError:
            pass
    if not valid:
        raise ValueError(
            "isolated PostgreSQL opt-in requires an explicit loopback port, "
            "a generated CAP005 database, and no connection-target overrides"
        ) from None


def is_local_pytest(command: Sequence[str]) -> bool:
    if not command:
        return False
    executable = Path(command[0]).name
    if executable in {"pytest", "py.test"}:
        return True
    if PYTHON_EXECUTABLE.fullmatch(executable) is None:
        return False
    index = 1
    while index < len(command):
        argument = command[index]
        if argument == "-m":
            return index + 1 < len(command) and command[index + 1] == "pytest"
        if argument.startswith("-m"):
            return argument == "-mpytest"
        if argument in {"-c", "--"} or not argument.startswith("-"):
            return False
        index += 2 if argument in {"-W", "-X"} else 1
    return False


@dataclass(frozen=True)
class FocusedPytest:
    command: list[str]
    environment: dict[str, str] = field(repr=False)
    root: Path
    report_path: Path
    source_identity: str

    def verify(
        self, completed: subprocess.CompletedProcess[str]
    ) -> subprocess.CompletedProcess[str]:
        if completed.returncode:
            return completed
        try:
            if evidence.source_identity(self.root) != self.source_identity:
                raise ValueError("source changed while focused pytest was running")
            receipt = evidence.load_pytest_receipt(
                self.report_path,
                root=self.root,
                expected_identity=self.source_identity,
                require_session=True,
            )
            if receipt.collection_skips:
                raise ValueError("focused pytest skipped a collector")
            if receipt.discovered_nodeids != frozenset(receipt.records):
                # Explicit nodeids discover sibling tests in the same module.
                selection = (receipt.session or {}).get("selection")
                targets = (
                    selection.get("targets") if isinstance(selection, dict) else None
                )
                if (
                    not isinstance(targets, list)
                    or not targets
                    or any(not isinstance(item, str) or not item for item in targets)
                ):
                    raise ValueError(
                        "focused pytest did not execute its full discovery"
                    )
                errors = receipt.complete_selection_errors(targets, root=self.root)
                if errors:
                    raise ValueError("; ".join(errors))
            if not all(
                isinstance(record, dict)
                and record.get("status") == "PASS"
                and record.get("phases") == PASSED_PHASES
                for record in receipt.records.values()
            ):
                raise ValueError("focused pytest has an unpassed test or phase")
        except (OSError, RuntimeError, ValueError) as exc:
            return subprocess.CompletedProcess(
                completed.args,
                1,
                completed.stdout,
                (completed.stderr or "")
                + f"\nfocused pytest evidence rejected: {exc}\n",
            )
        return completed


@contextmanager
def prepare_focused_pytest(
    command: Sequence[str],
    *,
    cwd: Path | None,
    environment: Mapping[str, str] | None,
    isolated_postgres_url: str | None = None,
) -> Iterator[FocusedPytest | None]:
    if not is_local_pytest(command):
        if isolated_postgres_url is not None:
            raise ValueError(
                "isolated PostgreSQL opt-in is only valid for direct local pytest"
            )
        yield None
        return
    if isolated_postgres_url is not None:
        validate_isolated_postgres_url(isolated_postgres_url)
    root = (cwd or Path.cwd()).resolve()
    identity = evidence.source_identity(root)
    with tempfile.TemporaryDirectory(prefix="gpu-fault-focused-pytest-") as directory:
        report_path = Path(directory) / "report.json"
        child_environment = build_isolated_environment(environment)
        if isolated_postgres_url is not None:
            child_environment["GPU_FAULT_TEST_POSTGRES_URL"] = isolated_postgres_url
        child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        child_environment["PYTEST_GPU_FAULT_CASE_REPORT"] = str(report_path)
        yield FocusedPytest(
            [*command, "-p", "tools.pytest_case_reporter", "-o", "addopts="],
            child_environment,
            root,
            report_path,
            identity,
        )
