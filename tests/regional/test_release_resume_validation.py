from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_resume_validation as RESUME

ROOT = Path(__file__).resolve().parents[2]


def _fixture(
    *, candidate_agent: str = "a" * 64, aurora_drift: Callable[[], bool] | None = None
):
    target = SimpleNamespace(cluster_id="gpu-a", context="gpu-a")
    previous_agent = "a" * 64
    previous_executor = "b" * 64
    metadata = {
        "required-agent-artifact-sha256": previous_agent,
        "required-agent-compatibility-digest": previous_agent,
        "required-agent-protocol-version": "3",
        "required-agent-config-digest": "c" * 64,
        "required-regional-executor-artifact-sha256": previous_executor,
        "required-regional-executor-compatibility-digest": previous_executor,
        "required-regional-executor-protocol-version": "2",
    }
    release = SimpleNamespace(
        config=SimpleNamespace(
            clusters=(target,),
            namespace="gpu-fault-system",
            agent_protocol_version=3,
            executor_protocol_version=2,
            component_digests={
                "node_runtime": candidate_agent,
                "executor": previous_executor,
            },
            agent_config_digest="c" * 64,
            runtime_profile_version="profile-a",
        ),
        cluster_registry_digest="registry-a",
        node_wheel_sha=candidate_agent,
        executor_wheel_sha=previous_executor,
        _config_map_data=lambda _name: dict(metadata),
        _target_node_names=lambda _target: ("node-a",),
        _aurora_refresh_drift=aurora_drift or (lambda: False),
    )
    previous = {
        "metadata": dict(metadata),
        "runtime_image": "runtime@sha256:" + "d" * 64,
        "agent_identities": {
            "gpu-a": {
                "node_ids": ["node-a"],
                "agent_protocol_version": 3,
                "artifact_sha256": previous_agent,
                "compatibility_digest": previous_agent,
                "installer_bundle_sha256": "e" * 64,
                "installer_template_sha256": "f" * 64,
                "runtime_profile_version": "profile-a",
                "config_digest": "c" * 64,
            }
        },
    }
    loaded = {
        "cluster_ids": ["gpu-a"],
        "cluster_registry_digest": "registry-a",
        "cluster_attempts": {"gpu-a": {"state": "PENDING", "attempt_generation": 1}},
        "completed_phases": [],
    }
    plan = RESUME.ReleaseExecutionPlan(nodes=(RESUME.ReleaseComponent.VERIFY,))
    return release, previous, loaded, plan, metadata


def test_resume_validator_accepts_unchanged_pending_cluster() -> None:
    release, previous, loaded, plan, _metadata = _fixture()

    RESUME.validate_resume_checkpoint(
        release, loaded=loaded, previous=previous, plan=plan
    )


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("drift", [False, True])
def test_resume_rechecks_completed_aurora_refresher_before_skipping_its_phase(
    completed: bool, drift: bool
) -> None:
    reads: list[bool] = []
    release, previous, loaded, _plan, _metadata = _fixture(
        aurora_drift=lambda: reads.append(True) or drift
    )
    if completed:
        loaded["completed_phases"] = ["aurora-refresh-ready"]
    plan = RESUME.ReleaseExecutionPlan(
        nodes=(RESUME.ReleaseComponent.AURORA_REFRESH, RESUME.ReleaseComponent.VERIFY)
    )
    if completed and drift:
        with pytest.raises(RESUME.ReleaseError, match="refresher drifted"):
            RESUME.validate_resume_checkpoint(
                release, loaded=loaded, previous=previous, plan=plan
            )
    else:
        RESUME.validate_resume_checkpoint(
            release, loaded=loaded, previous=previous, plan=plan
        )
    assert reads == ([True] if completed else [])


def test_resume_validator_rejects_node_set_drift() -> None:
    release, previous, loaded, plan, _metadata = _fixture()
    setattr(release, "_target_node_names", lambda _target: ("node-a", "node-b"))
    plan = RESUME.ReleaseExecutionPlan(
        nodes=(RESUME.ReleaseComponent.EXECUTOR, RESUME.ReleaseComponent.VERIFY)
    )

    with pytest.raises(RESUME.ReleaseError, match="node set drifted"):
        RESUME.validate_resume_checkpoint(
            release, loaded=loaded, previous=previous, plan=plan
        )


def test_resume_narrows_the_recorded_node_set_by_a_proven_departure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, previous, loaded, plan, _metadata = _fixture()
    previous["agent_identities"]["gpu-a"]["node_ids"] = ["node-a", "node-b"]
    asked = []

    def departed(_release, target, node_ids, *, phase, what, present_node_names=None):
        asked.append(
            (target.cluster_id, tuple(node_ids), phase, what, present_node_names)
        )
        return frozenset({"node-b"})

    monkeypatch.setattr(RESUME, "departed_node_ids", departed)
    validated = []
    monkeypatch.setattr(
        RESUME,
        "validate_gpu_rollback_target",
        lambda *_args, **_kwargs: validated.append(_kwargs["components"]),
    )
    plan = RESUME.ReleaseExecutionPlan(
        nodes=(RESUME.ReleaseComponent.EXECUTOR, RESUME.ReleaseComponent.VERIFY)
    )

    RESUME.validate_resume_checkpoint(
        release, loaded=loaded, previous=previous, plan=plan
    )

    assert asked == [
        (
            "gpu-a",
            ("node-a", "node-b"),
            "resume",
            "resume previous node set",
            ("node-a",),
        )
    ]
    assert validated == [frozenset({RESUME.ReleaseComponent.EXECUTOR})]


def test_resume_only_asks_about_departure_when_recorded_nodes_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, previous, loaded, plan, _metadata = _fixture()
    setattr(release, "_target_node_names", lambda _target: ("node-a", "node-b"))
    monkeypatch.setattr(
        RESUME,
        "departed_node_ids",
        lambda *_args, **_kwargs: pytest.fail("an extra node is drift, not departure"),
    )
    plan = RESUME.ReleaseExecutionPlan(
        nodes=(RESUME.ReleaseComponent.EXECUTOR, RESUME.ReleaseComponent.VERIFY)
    )

    with pytest.raises(RESUME.ReleaseError, match="node set drifted"):
        RESUME.validate_resume_checkpoint(
            release, loaded=loaded, previous=previous, plan=plan
        )


def test_resume_refuses_a_missing_node_whose_departure_is_unproven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, previous, loaded, plan, _metadata = _fixture()
    previous["agent_identities"]["gpu-a"]["node_ids"] = ["node-a", "node-b"]

    def unproven(*_args, **_kwargs):
        raise RESUME.ReleaseError("gpu-a resume previous node set names nodes ...")

    monkeypatch.setattr(RESUME, "departed_node_ids", unproven)
    plan = RESUME.ReleaseExecutionPlan(
        nodes=(RESUME.ReleaseComponent.EXECUTOR, RESUME.ReleaseComponent.VERIFY)
    )

    with pytest.raises(RESUME.ReleaseError, match="not provably departed|names nodes"):
        RESUME.validate_resume_checkpoint(
            release, loaded=loaded, previous=previous, plan=plan
        )


def test_cpu_only_resume_does_not_require_gpu_previous_identity() -> None:
    release, previous, loaded, _plan, _metadata = _fixture()
    previous["agent_identities"] = {}
    plan = RESUME.ReleaseExecutionPlan(
        nodes=(RESUME.ReleaseComponent.CPU_FINALIZE, RESUME.ReleaseComponent.VERIFY)
    )
    RESUME.validate_resume_checkpoint(
        release, loaded=loaded, previous=previous, plan=plan
    )


def test_resume_validator_accepts_staged_compatibility_window() -> None:
    candidate = "9" * 64
    release, previous, loaded, _plan, metadata = _fixture(candidate_agent=candidate)
    metadata["compatible-agent-artifact-sha256s"] = candidate
    metadata["compatible-agent-compatibility-digests"] = candidate
    setattr(release, "_config_map_data", lambda _name: dict(metadata))
    loaded["completed_phases"] = ["cpu-staged"]
    plan = RESUME.ReleaseExecutionPlan(
        nodes=(RESUME.ReleaseComponent.CPU_STAGE, RESUME.ReleaseComponent.VERIFY)
    )

    RESUME.validate_resume_checkpoint(
        release, loaded=loaded, previous=previous, plan=plan
    )


def test_resume_validator_accepts_promoted_window_before_checkpoint() -> None:
    """A finalize interrupted after promotion but before the checkpoint resumes.

    ``cpu-finalize`` promotes the pin window and only then runs the fleet
    barrier, so a barrier failure leaves the live window on the candidate with
    ``cpu-finalized`` unrecorded. Refusing that state wedges the transaction:
    resume is blocked and rollback is already impossible.
    """
    candidate = "9" * 64
    release, previous, loaded, _plan, metadata = _fixture(candidate_agent=candidate)
    metadata["required-agent-artifact-sha256"] = candidate
    metadata["required-agent-compatibility-digest"] = candidate
    setattr(release, "_config_map_data", lambda _name: dict(metadata))
    loaded["completed_phases"] = ["cpu-staged"]
    plan = RESUME.ReleaseExecutionPlan(
        nodes=(RESUME.ReleaseComponent.CPU_FINALIZE, RESUME.ReleaseComponent.VERIFY)
    )

    RESUME.validate_resume_checkpoint(
        release, loaded=loaded, previous=previous, plan=plan
    )


def test_resume_validator_rejects_pin_promoted_to_a_third_value() -> None:
    """Only the candidate is tolerated; an unrelated pin is still drift."""
    candidate = "9" * 64
    release, previous, loaded, _plan, metadata = _fixture(candidate_agent=candidate)
    metadata["required-agent-artifact-sha256"] = "7" * 64
    setattr(release, "_config_map_data", lambda _name: dict(metadata))
    loaded["completed_phases"] = ["cpu-staged"]
    plan = RESUME.ReleaseExecutionPlan(
        nodes=(RESUME.ReleaseComponent.CPU_FINALIZE, RESUME.ReleaseComponent.VERIFY)
    )

    with pytest.raises(RESUME.ReleaseError, match="staged required pin drifted"):
        RESUME.validate_resume_checkpoint(
            release, loaded=loaded, previous=previous, plan=plan
        )


def test_resume_validator_rejects_lost_candidate_pin() -> None:
    candidate = "9" * 64
    release, previous, loaded, _plan, _metadata = _fixture(candidate_agent=candidate)
    loaded["completed_phases"] = ["cpu-staged"]
    plan = RESUME.ReleaseExecutionPlan(
        nodes=(RESUME.ReleaseComponent.CPU_STAGE, RESUME.ReleaseComponent.VERIFY)
    )

    with pytest.raises(RESUME.ReleaseError, match="lost candidate"):
        RESUME.validate_resume_checkpoint(
            release, loaded=loaded, previous=previous, plan=plan
        )
