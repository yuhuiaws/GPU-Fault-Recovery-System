"""Consume fresh local pytest evidence without promoting it to LIVE evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from tools.pytest_result_identity import (
    PytestReceipt,
    normalized_pytest_nodeid,
    parse_pytest_receipt,
)
from tools.scenario_requirements import Check

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PytestEvidence:
    receipt: PytestReceipt
    root: Path

    def confirms(self, check: Check) -> bool:
        if self.receipt.discovered_nodeids is None or self.receipt.collection_skips:
            return False
        records, error = self.receipt.selection(
            normalized_pytest_nodeid(check.nodeid, root=self.root)
        )
        return (
            error is None
            and len(records) == check.expected
            and all(
                record.get("status") == "PASS"
                and record.get("phases")
                == {"setup": "passed", "call": "passed", "teardown": "passed"}
                for record in records
            )
        )


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("pytest evidence timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("pytest evidence timestamp must have a timezone")
    return parsed


def load_test_evidence(
    paths: Sequence[Path],
    *,
    expected_identity: str,
    now: datetime,
    max_age: timedelta = timedelta(days=7),
    root: Path = ROOT,
) -> PytestEvidence:
    discovered: set[str] = set()
    records: dict[str, object] = {}
    if now.tzinfo is None or max_age <= timedelta(0):
        raise ValueError("pytest evidence freshness bound is invalid")
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        receipt = parse_pytest_receipt(
            value, root=root, expected_identity=expected_identity, require_session=True
        )
        session = value["session"]
        if (
            session.get("collection_errors") != []
            or session.get("collection_skips") != []
        ):
            raise ValueError("pytest evidence lacks explicit clean collection metadata")
        started = _timestamp(session.get("started_at"))
        finished = _timestamp(session.get("finished_at"))
        if not now - max_age <= started <= finished <= now:
            raise ValueError("pytest evidence is stale or its time range is invalid")
        assert receipt.discovered_nodeids is not None
        for nodeid, record in receipt.records.items():
            if not isinstance(record, dict) or record.get("status") not in (
                "PASS",
                "FAIL",
            ):
                raise ValueError("pytest evidence result is invalid")
            if record["status"] == "PASS" and record.get("phases") != {
                "setup": "passed",
                "call": "passed",
                "teardown": "passed",
            }:
                raise ValueError("pytest PASS lacks completed setup, call and teardown")
            # A later PASS cannot hide a failure in another supplied shard.
            previous = records.get(nodeid)
            if previous is None or record["status"] == "FAIL":
                records[nodeid] = record
        discovered.update(receipt.discovered_nodeids)
    return PytestEvidence(PytestReceipt(records, frozenset(discovered)), root)
