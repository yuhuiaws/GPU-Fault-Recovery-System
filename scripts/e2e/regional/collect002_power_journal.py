"""COLLECT-002's power-operation journal: one single-use run id per attempt.

The host probe keeps its own journal of every run id it mutated under -- the
power record and the load-start receipt survive the cleanup and any reboot --
so a run id is single-use on the node (``power run ID has already been used;
cleanup only``). The runner derives the id from the run directory and the
attempt, which makes a retry of an attempt that never mutated the node
legitimate and a retry of one that did a deterministic refusal at the probe.
The runner's journal (``power-operation-a<attempt>.json``) records whether the
attempt mutated the node and how its restore ended, so the refusal is made here,
by name, before any probe runs (live HyperPod COLLECT-002: three attempts under
one id, two refused at ``throttle_gpu`` after the first restore failed).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.regional_commands import RegionalFixtureError

CASE_ID = "GF-REGIONAL-COLLECT-002"
CLEANUP_TOOL = "scripts/e2e/regional/collect002_power_cleanup.py"


def power_journal_path(case_dir: Path, attempt: int) -> Path:
    return case_dir / f"power-operation-a{attempt}.json"


def power_run_id(case_dir: Path, attempt: int) -> str:
    identity = f"{case_dir.resolve()}\0{attempt}".encode()
    return f"collect002-{hashlib.sha256(identity).hexdigest()[:24]}-a{attempt}"


def read_power_journal(case_dir: Path, attempt: int) -> dict[str, Any] | None:
    path = power_journal_path(case_dir, attempt)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RegionalFixtureError(f"{path.name} is not a JSON object")
    return value


def claim_power_run(case_dir: Path, attempt: int) -> str:
    """The attempt's run id, refused by name when the node already spent it.

    A previous execution of this attempt that reported ``mutation_started``, a
    restore failure (``cleanup_error``) or a restore that closed a record the
    probe had written (``power_record_seen``) spent the id on the node; this
    attempt cannot throttle under it again, and the refusal says so before the
    probe is asked. An execution refused before the probe wrote anything (none
    of those fields) may retry under the same id.
    """

    run_id = power_run_id(case_dir, attempt)
    previous = read_power_journal(case_dir, attempt)
    if previous is not None and (
        previous.get("mutation_started") is True
        or previous.get("power_record_seen") is True
        or previous.get("cleanup_error")
    ):
        failed_restore = bool(previous.get("cleanup_error"))
        raise RegionalFixtureError(
            f"attempt {attempt} of this run-dir already drove power operation "
            f"{run_id} on the node"
            + (
                f" and its restore failed ({previous['cleanup_error']})"
                if failed_restore
                else ""
            )
            + "; the node keeps a power run id single-use, so rerun with a new "
            "--attempt"
            + (
                f" after {CLEANUP_TOOL} has closed the old operation"
                if failed_restore
                else ""
            )
        )
    write_json_atomic(
        power_journal_path(case_dir, attempt),
        {"case_id": CASE_ID, "attempt": attempt, "run_id": run_id},
    )
    return run_id


def record_power_outcome(case_dir: Path, attempt: int, **fields: Any) -> None:
    """Merge what the probe reported into the attempt's journal entry."""

    current = read_power_journal(case_dir, attempt) or {
        "case_id": CASE_ID,
        "attempt": attempt,
        "run_id": power_run_id(case_dir, attempt),
    }
    write_json_atomic(power_journal_path(case_dir, attempt), {**current, **fields})
