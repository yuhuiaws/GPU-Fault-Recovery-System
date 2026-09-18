"""Small operation contracts at release phase boundaries.

The concrete release still owns configuration and state. Persistence documents
stay with their existing validators; these contracts do not duplicate their
schema or build another orchestration framework.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, Unpack, runtime_checkable

if TYPE_CHECKING:
    from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseConfig
    from gpu_fault_release.regional_release_diff import (
        ReleaseDiff,
        ReleaseExecutionPlan,
    )
    from gpu_fault_release.regional_release_gpu_rollout import ProgressCallback


class AgentNodeSet(TypedDict):
    node_ids: list[str]


class UpgradeOptions(TypedDict, total=False):
    resume: bool
    diff: ReleaseDiff | None
    supersede: dict[str, Any] | None


class RollbackOptions(TypedDict, total=False):
    state: dict[str, Any] | None
    automatic: bool


class GpuRolloutOptions(TypedDict, total=False):
    progress: ProgressCallback | None
    candidate_preflighted: bool


class RollbackTargetArguments(TypedDict):
    artifact: str
    config_digest: str
    runtime_profile_version: str
    executor_artifact: str
    executor_compatibility: str
    node_compatibility: str
    runtime_image: str
    node_installer_image: str


@runtime_checkable
class Snapshot(Protocol):
    @property
    def config(self) -> ReleaseConfig: ...

    @property
    def release_id(self) -> str: ...

    def _capture_previous(
        self, plan: ReleaseExecutionPlan | None = None
    ) -> dict[str, Any]: ...


@runtime_checkable
class CpuRollout(Protocol):
    def _apply_cpu(
        self,
        *,
        finalize: bool,
        force_restart: bool = False,
        diff: ReleaseDiff | None = None,
    ) -> None: ...

    def _capture_active_agent_node_sets(self) -> dict[str, AgentNodeSet]: ...

    def _candidate_agent_pin_identity(self) -> dict[str, str]: ...

    def _wait_candidate_cpu_agent_heartbeats(
        self,
        agent_identities: dict[str, Any],
        *,
        required_identity: dict[str, str] | None = None,
    ) -> None: ...


@runtime_checkable
class GpuRollout(Protocol):
    @property
    def config(self) -> ReleaseConfig: ...

    state: dict[str, Any]

    def _save_state(self, phase: str, **updates: object) -> None: ...

    def _upgrade_gpu_target(
        self,
        target: ClusterTarget,
        diff: ReleaseDiff,
        plan: ReleaseExecutionPlan | None = None,
        **options: Unpack[GpuRolloutOptions],
    ) -> None: ...


class RollbackCheckpoint(Protocol):
    def __call__(
        self,
        phase: str,
        *,
        timing_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None: ...


RollbackPhaseRunner = Callable[[str, str, str, Callable[[], object]], None]
