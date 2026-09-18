"""One execution per approved run; an existing attempt grants cleanup only."""

from __future__ import annotations

from collections.abc import Callable
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

    def _read(self) -> ExecutionRecord:
        try:
            record = ExecutionRecord.model_validate(read_private_document(self.path))
        except (OSError, ValueError, TypeError):
            raise RegionalFixtureError("execution journal is invalid") from None
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
