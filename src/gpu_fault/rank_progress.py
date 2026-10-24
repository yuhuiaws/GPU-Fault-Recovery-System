"""Optional application-owned, non-collective progress snapshot protocol."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from gpu_fault.models import StrictModel


class ProcessGroupProgress(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    members: list[int] = Field(min_length=1, max_length=65536)
    enqueued: int = Field(ge=-1)
    started: int = Field(ge=-1)
    completed: int = Field(ge=-1)

    @field_validator("members")
    @classmethod
    def unique_members(cls, value: list[int]) -> list[int]:
        if any(rank < 0 for rank in value) or len(set(value)) != len(value):
            raise ValueError(
                "process group membership must be unique nonnegative ranks"
            )
        return sorted(value)

    def descriptor(self) -> tuple[str, dict[str, Any]]:
        digest = hashlib.sha256(
            json.dumps(self.members, separators=(",", ":")).encode()
        ).hexdigest()
        key = hashlib.sha256(f"{self.name}/{digest}".encode()).hexdigest()
        return key, {
            "name": self.name,
            "member_count": len(self.members),
            "membership_sha256": digest,
        }


class DiagnosticContext(StrictModel):
    group_name: str = Field(min_length=1, max_length=128)
    members: list[int] = Field(min_length=1, max_length=65536)
    role: str = Field(min_length=1, max_length=128)
    pipeline_stage: str = Field(min_length=1, max_length=128)
    expert_role: str = Field(min_length=1, max_length=128)
    hardware_class: str = Field(min_length=1, max_length=128)
    code_version: str = Field(min_length=1, max_length=128)
    thread_roles: dict[str, Literal["TRAINING", "COMMUNICATION", "BACKGROUND"]] = Field(
        default_factory=dict, max_length=64
    )

    @model_validator(mode="after")
    def validate_identity(self) -> DiagnosticContext:
        if any(type(rank) is not int or rank < 0 for rank in self.members) or len(
            set(self.members)
        ) != len(self.members):
            raise ValueError(
                "diagnostic group members must be unique nonnegative ranks"
            )
        if any(not tid.isdecimal() or int(tid) <= 0 for tid in self.thread_roles):
            raise ValueError("thread roles require native thread IDs")
        return self


class RankProgressSnapshot(StrictModel):
    version: Literal["rank-progress/v1"] = "rank-progress/v1"
    job_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    rank: int = Field(ge=0)
    world_size: int = Field(ge=1, le=65536)
    process_start_ticks: int = Field(gt=0)
    process_pid: int | None = Field(default=None, gt=0)
    observed_at: datetime
    step: int | None = Field(default=None, ge=0)
    phase: Literal[
        "TRAINING",
        "INITIALIZING",
        "CHECKPOINT_LOAD",
        "CHECKPOINT_SAVE",
        "RESTORING",
        "UNKNOWN",
    ] = "UNKNOWN"
    process_groups: list[ProcessGroupProgress] = Field(
        default_factory=list, max_length=64
    )
    diagnostic_context: DiagnosticContext | None = None

    @model_validator(mode="after")
    def identity(self) -> RankProgressSnapshot:
        if self.observed_at.tzinfo is None or self.rank >= self.world_size:
            raise ValueError("invalid progress timestamp or world rank")
        if any(
            self.rank not in group.members or max(group.members) >= self.world_size
            for group in self.process_groups
        ):
            raise ValueError("rank must belong to each declared process group")
        if len({group.descriptor()[0] for group in self.process_groups}) != len(
            self.process_groups
        ):
            raise ValueError("duplicate process group")
        if self.diagnostic_context and (
            self.rank not in self.diagnostic_context.members
            or max(self.diagnostic_context.members) >= self.world_size
        ):
            raise ValueError("diagnostic group membership does not match rank identity")
        return self


def write_rank_progress(
    path: str | Path | None = None,
    *,
    job_id: str,
    attempt_id: str,
    rank: int,
    world_size: int,
    step: int | None = None,
    phase: str = "TRAINING",
    process_groups: list[dict[str, Any]] | None = None,
    diagnostic_context: dict[str, Any] | None = None,
) -> None:
    """Write locally from a training process; never execute a collective or CUDA API."""
    stat_text = Path("/proc/self/stat").read_text()
    document = RankProgressSnapshot.model_validate(
        {
            "job_id": job_id,
            "attempt_id": attempt_id,
            "rank": rank,
            "world_size": world_size,
            "step": step,
            "phase": phase,
            "process_groups": process_groups or [],
            "diagnostic_context": diagnostic_context,
            "process_start_ticks": int(
                stat_text[stat_text.rfind(")") + 2 :].split()[19]
            ),
            "process_pid": os.getpid(),
            "observed_at": datetime.now(timezone.utc),
        }
    )
    selected = path or os.getenv("GPU_FAULT_RANK_PROGRESS_FILE", "")
    if not selected:
        raise ValueError("a rank progress path must be configured")
    target = Path(selected)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=target.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(document.model_dump_json())
            stream.flush()
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
