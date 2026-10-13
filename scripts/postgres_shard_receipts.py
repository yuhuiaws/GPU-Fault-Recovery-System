"""Validate local serial partitions without creating a CI aggregate receipt."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from gpu_fault.admin.command_log import _LONG_TOKEN
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.postgres_grant import private_directory
from tools.pytest_result_identity import (
    parse_pytest_receipt,
    partition_for_nodeid,
)

MAX_RECEIPT_BYTES = 32 * 1024 * 1024
# A worker's own non-test failure: how far it got and the exception chain.
WORKER_PHASES = frozenset({"preparation", "allocation", "pytest", "post-test"})
FAILURE_CHAIN_LIMIT = 4
FAILURE_MESSAGE_CHARACTERS = 500
_EXCEPTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


class PostgresShardError(RuntimeError):
    """Only controlled, credential-free messages may cross the runner boundary."""


def redact_output(text: str, *, limit: int) -> str:
    """``text`` as an operator may read a supervised command's output.

    ``diagnostic_text`` masks URL credentials, secret assignments, bearer
    tokens, private keys, access key IDs and control characters. The long-token
    mask is the one ``CommandRunner`` applies to a sensitive command's stderr;
    it also covers a generated database password inside PGPASSFILE content.
    """
    return _LONG_TOKEN.sub("<redacted>", diagnostic_text(text, limit=limit))


def failure_message(text: str) -> str:
    """One printable, redacted line of at most ``FAILURE_MESSAGE_CHARACTERS``."""
    line = "".join(
        character
        for character in " ".join(redact_output(text, limit=64 * 1024).split())
        if character.isprintable()
    )
    if len(line) <= FAILURE_MESSAGE_CHARACTERS:
        return line
    half = FAILURE_MESSAGE_CHARACTERS // 2 - 3
    return f"{line[:half]} ... {line[-half:]}"


def exception_chain(error: BaseException, *, limit: int) -> list[BaseException]:
    """``error`` and what it was raised from, outermost first.

    The links traceback follows: an explicit cause, else an unsuppressed context.
    """
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and len(chain) < limit:
        chain.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            current = None
        else:
            current = current.__context__
    return chain


def failure_chain(error: BaseException) -> list[dict[str, str]]:
    """What a worker records of ``error``: each link's type and redacted message."""
    chain: list[dict[str, str]] = []
    for link in exception_chain(error, limit=FAILURE_CHAIN_LIMIT):
        notes = getattr(link, "__notes__", ())
        text = "; ".join(part for part in (str(link), *notes) if part)
        chain.append({"type": type(link).__name__, "message": failure_message(text)})
    return chain


def describe_failure(errors: Sequence[Mapping[str, str]]) -> str:
    return "; caused by ".join(
        f"{item['type']}: {failure_message(item['message'])}"
        if item["message"]
        else item["type"]
        for item in errors
    )


def read_worker_failure(path: Path) -> dict[str, Any]:
    value = read_private_json(path, maximum=64 * 1024)
    errors = value.get("errors")
    if (
        set(value) != {"phase", "errors"}
        or not isinstance(value["phase"], str)
        or value["phase"] not in WORKER_PHASES
        or not isinstance(errors, list)
        or not 0 < len(errors) <= FAILURE_CHAIN_LIMIT
        or not all(
            isinstance(item, dict)
            and set(item) == {"type", "message"}
            and isinstance(item["type"], str)
            and _EXCEPTION_NAME.fullmatch(item["type"]) is not None
            and isinstance(item["message"], str)
            and len(item["message"]) <= FAILURE_MESSAGE_CHARACTERS
            and item["message"].isprintable()
            for item in errors
        )
    ):
        raise PostgresShardError("PostgreSQL worker failure record is invalid")
    return value


def safe_failure_nodeid(nodeid: object) -> bool:
    if not isinstance(nodeid, str) or not 0 < len(nodeid) <= 4096:
        return False
    filename = nodeid.partition("::")[0]
    path = PurePosixPath(filename)
    return (
        nodeid.isprintable()
        and bool(path.parts)
        and path.parts[0] == "tests"
        and path.suffix == ".py"
        and path.as_posix() == filename
        and ".." not in path.parts
    )


def read_failure(path: Path) -> dict[str, Any]:
    value = read_private_json(path, maximum=16 * 1024)
    if (
        set(value) != {"failed", "nodeid", "phase", "outcome"}
        or value["failed"] is not True
        or value["nodeid"] is not None
        and not safe_failure_nodeid(value["nodeid"])
        or not isinstance(value["phase"], str)
        or value["phase"] not in {"collect", "setup", "call", "teardown", "process"}
        or not isinstance(value["outcome"], str)
        or value["outcome"] not in {"failed", "skipped"}
    ):
        raise PostgresShardError("PostgreSQL first-failure metadata is invalid")
    return value


def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PostgresShardError("local PostgreSQL JSON has duplicate fields")
        result[key] = value
    return result


def read_private_json(
    path: Path, *, maximum: int = MAX_RECEIPT_BYTES
) -> dict[str, Any]:
    private_directory(path.parent)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            before = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or not 0 < before.st_size <= maximum
            ):
                raise PostgresShardError("local PostgreSQL JSON is not private")
            raw = source.read(maximum + 1)
            after = os.fstat(source.fileno())
            if len(raw) > maximum or (
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise PostgresShardError("local PostgreSQL JSON changed while reading")
        value = json.loads(raw, object_pairs_hook=unique_fields)
    except (OSError, ValueError):
        raise PostgresShardError(
            "local PostgreSQL JSON is missing or invalid"
        ) from None
    if not isinstance(value, dict):
        raise PostgresShardError("local PostgreSQL JSON must be an object")
    return value


def validate_targets(root: Path, tests: Sequence[str]) -> tuple[str, ...]:
    """Make supplies complete files; options and partial selectors are not targets."""
    if not tests or len(tests) != len(set(tests)):
        raise PostgresShardError("PostgreSQL targets must be nonempty and unique")
    for target in tests:
        path = Path(target)
        if (
            path.is_absolute()
            or path.as_posix() != target
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != "tests"
            or path.suffix != ".py"
            or "::" in target
            or not (root / path).is_file()
            or (root / path).resolve() != root / path
        ):
            raise PostgresShardError(
                "PostgreSQL targets must be canonical checkout test files"
            )
    return tuple(sorted(tests))


@dataclass(frozen=True)
class ShardReceipt:
    index: int
    discovered: frozenset[str]
    durations: dict[str, float]
    wall_seconds: float


def validate_shard_receipt(
    path: Path,
    *,
    root: Path,
    identity: str,
    tests: tuple[str, ...],
    workers: int,
    index: int,
    stress: tuple[str, str],
    started_after: datetime,
) -> ShardReceipt:
    value = read_private_json(path)
    try:
        receipt = parse_pytest_receipt(
            value,
            root=root,
            expected_identity=identity,
            require_session=True,
            require_passed=True,
        )
    except (TypeError, ValueError):
        raise PostgresShardError(
            "PostgreSQL shard lacks a complete source-bound passing receipt"
        ) from None
    session = receipt.session
    if session is None or receipt.discovered_nodeids is None:
        raise PostgresShardError("PostgreSQL shard session is missing")
    selection = session.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("targets") != list(tests)
        or selection.get("partition") != [workers, index]
        or any(type(item) is not int for item in selection.get("partition", []))
        or selection.get("keyword") != ""
        or selection.get("markexpr") != ""
        or selection.get("deselect") != []
        or type(selection.get("numprocesses")) is not int
        or selection["numprocesses"] != 0
        or type(selection.get("requested_numprocesses")) is not int
        or selection["requested_numprocesses"] != 0
        or (selection.get("stress_workers"), selection.get("stress_rounds")) != stress
        or "ci_context" in session
    ):
        raise PostgresShardError("PostgreSQL shard selection differs from its request")
    if (
        session.get("collection_errors") != []
        or session.get("collection_skips") != []
        or session.get("collected_files") != list(tests)
    ):
        raise PostgresShardError("PostgreSQL shard did not collect every supplied file")
    try:
        started = datetime.fromisoformat(session["started_at"])
        finished = datetime.fromisoformat(session["finished_at"])
        valid_times = (
            started.tzinfo is not None
            and finished.tzinfo is not None
            and started_after <= started <= finished <= datetime.now(timezone.utc)
        )
    except (KeyError, TypeError, ValueError):
        valid_times = False
    if not valid_times:
        raise PostgresShardError("PostgreSQL shard timestamps are invalid or stale")

    def relative(nodeid: str) -> str:
        filename, separator, selector = nodeid.partition("::")
        try:
            target = Path(filename).relative_to(root).as_posix()
        except ValueError:
            raise PostgresShardError(
                "PostgreSQL discovery leaves the checkout"
            ) from None
        if target not in tests or not separator or not selector:
            raise PostgresShardError(
                "PostgreSQL discovery contains an unrequested file"
            )
        return target + separator + selector

    discovered = frozenset(relative(item) for item in receipt.discovered_nodeids)
    if {item.partition("::")[0] for item in discovered} != set(tests):
        raise PostgresShardError("PostgreSQL discovery omits a supplied test file")
    expected = {
        nodeid
        for nodeid in discovered
        if partition_for_nodeid(nodeid, workers) == index
    }
    durations: dict[str, float] = {}
    for nodeid, record in receipt.records.items():
        duration = record.get("duration_seconds") if isinstance(record, dict) else None
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration < 0
            or not isinstance(record, dict)
            or record.get("output") != ""
        ):
            raise PostgresShardError(
                "PostgreSQL shard duration/output evidence is invalid"
            )
        durations[relative(nodeid)] = float(duration)
    if not expected or set(durations) != expected:
        raise PostgresShardError(
            "PostgreSQL shard partition is empty or incomplete; use fewer workers "
            "for a small smoke selection"
        )
    return ShardReceipt(
        index, discovered, durations, (finished - started).total_seconds()
    )


def validate_partition_union(
    receipts: Sequence[ShardReceipt], *, workers: int
) -> dict[str, float]:
    if len(receipts) != workers or {item.index for item in receipts} != set(
        range(workers)
    ):
        raise PostgresShardError("PostgreSQL shard receipt inventory is incomplete")
    discovered = receipts[0].discovered
    durations: dict[str, float] = {}
    for receipt in receipts:
        if receipt.discovered != discovered:
            raise PostgresShardError("PostgreSQL shards disagree on full discovery")
        if durations.keys() & receipt.durations.keys():
            raise PostgresShardError("PostgreSQL shard partitions overlap")
        durations.update(receipt.durations)
    if durations.keys() != discovered:
        raise PostgresShardError("PostgreSQL shard union does not cover full discovery")
    return durations
