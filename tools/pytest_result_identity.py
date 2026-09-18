from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

CI_CONTEXT_ENV = "PYTEST_GPU_FAULT_CI_CONTEXT"
PASSED_PHASES = {"setup": "passed", "call": "passed", "teardown": "passed"}


def partition_for_nodeid(nodeid: str, count: int) -> int:
    if count < 1:
        raise ValueError("pytest partition count must be positive")
    digest = hashlib.sha256(nodeid.encode()).digest()
    return int.from_bytes(digest[:8], "big") % count


def passed_pytest_record(record: object) -> bool:
    return (
        isinstance(record, dict)
        and record.get("status") == "PASS"
        and record.get("phases") == PASSED_PHASES
    )


def normalized_pytest_nodeid(value: str, *, root: Path) -> str:
    path_text, separator, selector = value.partition("::")
    path = Path(path_text)
    resolved = (path if path.is_absolute() else root / path).resolve()
    return str(resolved) + (f"::{selector}" if separator else "")


@dataclass(frozen=True)
class PytestReceipt:
    records: dict[str, object]
    discovered_nodeids: frozenset[str] | None = None
    collection_skips: tuple[str, ...] = ()
    session: dict[str, Any] | None = None

    def selection(self, selector: str) -> tuple[list[dict[str, Any]], str | None]:
        def matches(nodeid: str) -> bool:
            return (
                nodeid == selector
                or nodeid.startswith(selector + "::")
                or (
                    "[" not in selector.partition("::")[2]
                    and nodeid.startswith(selector + "[")
                )
            )

        if self.discovered_nodeids is None:
            # Legacy local reports have no discovery evidence.
            exact = self.records.get(selector)
            if isinstance(exact, dict):
                return [exact], None
            return [
                record
                for nodeid, record in self.records.items()
                if matches(nodeid) and isinstance(record, dict)
            ], None
        expected = sorted(
            nodeid for nodeid in self.discovered_nodeids if matches(nodeid)
        )
        records = [
            record
            for nodeid in expected
            if isinstance(record := self.records.get(nodeid), dict)
        ]
        missing = [
            nodeid
            for nodeid in expected
            if not isinstance(self.records.get(nodeid), dict)
        ]
        if not expected:
            return records, f"pytest discovery contains no test for {selector}"
        if missing:
            return records, (
                f"pytest did not execute all discovered variants for {selector}; "
                f"missing={missing[:20]} (total {len(missing)})"
            )
        return records, None

    def complete_selection_errors(
        self, selectors: list[str], *, root: Path
    ) -> list[str]:
        """A local contract must execute every requested variant, without filters."""
        if self.discovered_nodeids is None or self.session is None:
            return [
                "pytest contract requires a fresh complete discovery/session receipt"
            ]
        selection = self.session.get("selection")
        if not isinstance(selection, dict):
            return ["pytest contract selection receipt is missing"]
        targets = selection.get("targets")
        expected = [normalized_pytest_nodeid(item, root=root) for item in selectors]
        if (
            not isinstance(targets, list)
            or any(not isinstance(item, str) for item in targets)
            or len(targets) != len(set(targets))
            or sorted(normalized_pytest_nodeid(item, root=root) for item in targets)
            != sorted(expected)
        ):
            return ["pytest contract targets differ from the required selectors"]
        if any(selection.get(key) for key in ("keyword", "markexpr", "deselect")) or (
            selection.get("partition") is not None
        ):
            return ["pytest contract was filtered or partitioned"]
        if (
            type(selection.get("numprocesses")) is not int
            or selection["numprocesses"] != 0
        ):
            return ["pytest contract requires its explicitly serial worker receipt"]
        try:
            started = datetime.fromisoformat(self.session["started_at"])
            finished = datetime.fromisoformat(self.session["finished_at"])
        except (KeyError, TypeError, ValueError):
            return ["pytest contract execution timestamps are missing or invalid"]
        if started.tzinfo is None or finished.tzinfo is None or finished < started:
            return ["pytest contract execution timestamps are unordered or unzoned"]
        errors = []
        if self.collection_skips:
            errors.append("pytest contract skipped a collector")
        wanted: set[str] = set()
        for selector in expected:
            records, error = self.selection(selector)
            if error is not None:
                errors.append(error)
            if not records or not all(passed_pytest_record(item) for item in records):
                errors.append(
                    f"pytest contract lacks successful setup/call/teardown: {selector}"
                )
            wanted.update(
                nodeid
                for nodeid in self.discovered_nodeids
                if nodeid == selector
                or nodeid.startswith(selector + "::")
                or (
                    "[" not in selector.partition("::")[2]
                    and nodeid.startswith(selector + "[")
                )
            )
        if set(self.records) != wanted:
            errors.append("pytest contract execution inventory differs from discovery")
        return errors


def _receipt_nodeids(value: object, *, label: str) -> set[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(nodeid, str) or "::" not in nodeid for nodeid in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"pytest {label} inventory is invalid")
    return set(value)


AggregateReceiptParser = Callable[[Mapping[str, Any], Path, str], PytestReceipt]


def parse_pytest_receipt(
    value: object,
    *,
    root: Path,
    expected_identity: str,
    require_session: bool = False,
    expected_exitstatus: int = 0,
    require_passed: bool = False,
    aggregate_parser: AggregateReceiptParser | None = None,
) -> PytestReceipt:
    """Validate a decoded receipt without reading its file a second time."""

    if isinstance(value, dict) and value.get("schema_version") == 2:
        if require_session or expected_exitstatus != 0:
            raise ValueError("a CI aggregate is not a fresh pytest session")
        if aggregate_parser is None:
            raise ValueError("a CI aggregate requires its content-identity validator")
        return aggregate_parser(value, root, expected_identity)
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("source_identity") != expected_identity
        or not isinstance(value.get("records"), dict)
    ):
        raise ValueError("pytest case result identity does not match current source")
    if any(
        not isinstance(nodeid, str) or "::" not in nodeid for nodeid in value["records"]
    ):
        raise ValueError("pytest result nodeid inventory is invalid")
    if require_passed and (
        not value["records"]
        or any(not passed_pytest_record(record) for record in value["records"].values())
    ):
        raise ValueError("pytest PASS requires completed setup, call and teardown")
    discovered = None
    skips: list[str] = []
    if require_session or "session" in value:
        session = value.get("session")
        if (
            not isinstance(session, dict)
            or session.get("source_identity") != expected_identity
            or type(session.get("exitstatus")) is not int
            or session["exitstatus"] != expected_exitstatus
        ):
            raise ValueError(
                "pytest case result session is incomplete or source changed"
            )
        try:
            collected = _receipt_nodeids(
                session.get("collected_nodeids"), label="collection"
            )
        except ValueError as exc:
            raise ValueError(
                "pytest case results do not cover the complete collection"
            ) from exc
        if collected != set(value["records"]):
            raise ValueError("pytest case results do not cover the complete collection")
        discovered = _receipt_nodeids(
            session.get("discovered_nodeids"), label="discovery"
        )
        if not collected <= discovered:
            raise ValueError("pytest discovery omits collected test results")
        errors = session.get("collection_errors", [])
        if not isinstance(errors, list) or errors:
            raise ValueError("pytest discovery contains collection errors")
        skips = session.get("collection_skips", [])
        if not isinstance(skips, list) or any(
            not isinstance(nodeid, str) for nodeid in skips
        ):
            raise ValueError("pytest skipped collector inventory is invalid")
    records = {
        normalized_pytest_nodeid(nodeid, root=root): record
        for nodeid, record in value["records"].items()
    }
    if discovered is not None and len(records) != len(value["records"]):
        raise ValueError("pytest result paths alias the same collected test")
    normalized_discovery = (
        frozenset(normalized_pytest_nodeid(nodeid, root=root) for nodeid in discovered)
        if discovered is not None
        else None
    )
    if (
        discovered is not None
        and normalized_discovery is not None
        and len(discovered) != len(normalized_discovery)
    ):
        raise ValueError("pytest discovery paths alias the same test")
    return PytestReceipt(
        records, normalized_discovery, tuple(skips), value.get("session")
    )


def load_pytest_receipt(
    path: Path,
    *,
    root: Path,
    expected_identity: str,
    require_session: bool = False,
    expected_exitstatus: int = 0,
    require_passed: bool = False,
    aggregate_parser: AggregateReceiptParser | None = None,
) -> PytestReceipt:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"pytest case results are invalid: {path}") from exc
    return parse_pytest_receipt(
        value,
        root=root,
        expected_identity=expected_identity,
        require_session=require_session,
        expected_exitstatus=expected_exitstatus,
        require_passed=require_passed,
        aggregate_parser=aggregate_parser,
    )


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "git command failed while identifying pytest results: "
            + completed.stderr.decode(errors="replace").strip()
        )
    return completed.stdout


def repository_root(path: Path) -> Path:
    value = os.fsdecode(
        _git_bytes(path.resolve(), "rev-parse", "--show-toplevel")
    ).removesuffix("\n")
    root = Path(value)
    if not root.is_absolute():
        raise RuntimeError("git did not identify an absolute repository root")
    return root.resolve()


def source_identity(root: Path) -> str:
    root = root.resolve()
    digest = hashlib.sha256()
    digest.update(_git_bytes(root, "rev-parse", "HEAD").strip())
    digest.update(b"\0diff\0")
    digest.update(_git_bytes(root, "diff", "--binary", "HEAD"))
    untracked = [
        Path(os.fsdecode(value))
        for value in _git_bytes(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).split(b"\0")
        if value
    ]
    for relative in sorted(untracked, key=lambda value: value.as_posix()):
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"untracked pytest input leaves repository: {relative}")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"unsupported untracked pytest input: {relative}")
        digest.update(b"\0untracked\0")
        digest.update(relative.as_posix().encode())
        digest.update(f"{path.stat().st_mode & 0o777:04o}".encode())
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
