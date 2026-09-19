"""One execution per approved run; an existing attempt grants cleanup only."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional.destr008_controller_lock import (
    controller_ownership,
    read_private_document,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError

Stage = Literal["workload_started", "safety_started", "fixture_started", "post_started"]
SCENARIOS = (
    "no-spare",
    "topology-mismatch",
    "kubernetes-not-ready",
    "reserved-by-other",
    "active-gpu-pod",
    "agent-unavailable",
)


class ScenarioState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    state: Literal["PENDING", "STARTED", "CLEANED"] = "PENDING"
    workload_started: bool = False
    safety_started: bool = False
    fixture_started: bool = False
    post_started: bool = False

    @model_validator(mode="after")
    def lifecycle(self) -> ScenarioState:
        if self.state == "PENDING" and any(
            (
                self.workload_started,
                self.safety_started,
                self.fixture_started,
                self.post_started,
            )
        ):
            raise ValueError("pending scenario cannot own started resources")
        return self


class ExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = Field(ge=1, le=1)
    binding: dict[str, Any]
    attempt: int = Field(ge=1)
    maintenance_expires_at: int = Field(gt=0)
    release_id: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    fault_uid: str = Field(min_length=1)
    spare_uid: str = Field(min_length=1)
    scenarios: dict[str, ScenarioState]
    prewarm_started: bool = False
    prewarm_cleaned: bool = False
    completed: bool = False

    @model_validator(mode="after")
    def lifecycle(self) -> ExecutionRecord:
        inputs = self.binding.get("inputs")
        approved = inputs.get("scenarios") if isinstance(inputs, dict) else None
        if (
            not isinstance(approved, list)
            or not approved
            or any(
                not isinstance(name, str) or name not in SCENARIOS for name in approved
            )
            or len(set(approved)) != len(approved)
            or set(self.scenarios) != set(approved)
            or sum(item.state == "STARTED" for item in self.scenarios.values()) > 1
            or (
                not self.prewarm_started
                and any(item.state == "STARTED" for item in self.scenarios.values())
            )
            or (
                self.completed
                and (
                    not self.prewarm_cleaned
                    or any(item.state == "STARTED" for item in self.scenarios.values())
                )
            )
        ):
            raise ValueError("execution journal lifecycle is invalid")
        return self


class ExecutionJournal:
    def __init__(
        self,
        path: Path,
        binding: Callable[[], dict[str, Any]],
        *,
        initial: ExecutionRecord | None = None,
    ) -> None:
        self.path, self.binding = path, binding
        with controller_ownership(path):
            self.resuming = path.exists() or path.is_symlink()
            if self.resuming:
                self.record = self._read()
            elif initial is None:
                raise RegionalFixtureError("original execution journal is missing")
            else:
                self.record = initial
                if initial.binding != binding():
                    raise RegionalFixtureError(
                        "execution journal initial binding differs"
                    )
                self._save()

    @staticmethod
    def peek(path: Path) -> ExecutionRecord:
        """The validated record at ``path``, without comparing its binding.

        Callers hold ``controller_ownership`` for the journal. This only says
        what the document records, never whether the caller may act on it.
        """
        try:
            return ExecutionRecord.model_validate(read_private_document(path))
        except (OSError, ValueError, TypeError):
            raise RegionalFixtureError("execution journal is invalid") from None

    def _read(self) -> ExecutionRecord:
        record = self.peek(self.path)
        if record.binding != self.binding():
            raise RegionalFixtureError("execution journal input or connection drifted")
        return record

    def _save(self) -> None:
        data = self.record.model_dump(mode="json")
        ExecutionRecord.model_validate(data)
        write_json_atomic(self.path, data)

    def start_prewarm(self) -> None:
        with controller_ownership(self.path):
            self.record = self._read()
            if self.resuming or self.record.prewarm_started or self.record.completed:
                raise RegionalFixtureError("existing execution is cleanup-only")
            self.record.prewarm_started = True
            self._save()

    def start_scenario(self, scenario: str) -> None:
        with controller_ownership(self.path):
            self.record = self._read()
            if (
                self.resuming
                or self.record.completed
                or not self.record.prewarm_started
                or scenario not in self.record.scenarios
                or self.record.scenarios[scenario].state != "PENDING"
                or any(
                    item.state == "STARTED" for item in self.record.scenarios.values()
                )
            ):
                raise RegionalFixtureError("scenario cannot be started or repeated")
            self.record.scenarios[scenario].state = "STARTED"
            self._save()

    def stage(self, scenario: str, stage: Stage) -> None:
        with controller_ownership(self.path):
            self.record = self._read()
            current = self.record.scenarios.get(scenario)
            if self.resuming or current is None or current.state != "STARTED":
                raise RegionalFixtureError("scenario has no execution authority")
            if getattr(current, stage):
                raise RegionalFixtureError("scenario stage cannot be repeated")
            setattr(current, stage, True)
            self._save()

    def cleaned_scenario(self, scenario: str) -> None:
        with controller_ownership(self.path):
            self.record = self._read()
            current = self.record.scenarios.get(scenario)
            if current is None or current.state not in {"STARTED", "CLEANED"}:
                raise RegionalFixtureError("scenario has no cleanup intent")
            current.state = "CLEANED"
            self._save()

    def cleaned_prewarm(self) -> None:
        with controller_ownership(self.path):
            self.record = self._read()
            self.record.prewarm_cleaned = True
            self._save()

    def complete(self) -> None:
        with controller_ownership(self.path):
            self.record = self._read()
            self.record.completed = True
            self._save()


def owns_nothing(record: ExecutionRecord) -> bool:
    return (
        record.completed
        and record.prewarm_cleaned
        and not any(item.state == "STARTED" for item in record.scenarios.values())
    )


def _archive_name(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def retire_completed_execution(
    case_dir: Path,
    journal_path: Path,
    *,
    binding: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Archive a finished execution so the next attempt can start fresh.

    A record that is ``completed`` with its prewarm cleaned and no scenario
    STARTED owns no cluster resource, and it grants no re-execution either: an
    ``ExecutionJournal`` opened on it only ever answers cleanup-only, and its
    binding check fails outright once a deploy rewrote a kubeconfig. So the
    journal, its lock, ``scenarios/`` and the prewarm owner move to
    ``.completed-<stamp>`` names beside the originals and the lineage is
    returned. Anything unfinished stays exactly in place (``None``) for the
    strict, binding-checked resume path.
    """

    if not _present(journal_path):
        return None
    with controller_ownership(journal_path):
        if not _present(journal_path):
            return None
        record = ExecutionJournal.peek(journal_path)
        if not owns_nothing(record):
            return None
        matched = None if binding is None else record.binding == binding()
        # Siblings first, the journal after them, its lock last: a crash in
        # between leaves a still-finished journal that the next attempt retires
        # again, never ownerless artifacts that would demand reconciliation.
        sources = (
            case_dir / "scenarios",
            case_dir / "prewarm-owner.json",
            case_dir / "prewarm-owner.lock",
            journal_path,
            journal_path.with_suffix(".lock"),
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag, collisions = f"completed-{stamp}", 0
        while any(_present(_archive_name(source, tag)) for source in sources):
            collisions += 1
            tag = f"completed-{stamp}-{collisions}"
        archived: dict[str, str] = {}
        for source in sources:
            if _present(source):
                target = _archive_name(source, tag)
                source.rename(target)
                archived[source.name] = target.name
    return {
        "attempt": record.attempt,
        "release_id": record.release_id,
        "binding_matched": matched,
        "archived": archived,
    }
